import tensorflow as tf
from layers import Layer


class Model:
    def _build_model(self, inputs, layers, num_classes, is_training):
        raise NotImplementedError

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):
            """
            Estimator's model_fn
            :param features:
            :param labels:
            :param mode: Specifies if this training, evaluation or prediction. See `ModeKeys`.
            :param params: net, image_size, num_classes
            :return: EstimatorSpec
            """
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            logits = self._build_model(layers=params['net'],
                                       num_classes=params['num_classes'],
                                       inputs=features,
                                       is_training=is_training)
            predicted_classes = tf.argmax(logits, 1)
            predictions = {
                'class_ids': predicted_classes,
                'probabilities': tf.nn.softmax(logits)
            }
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) \
                   + tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes)
            eval_metric_ops = {
                'accuracy': accuracy
            }
            return tf.estimator.EstimatorSpec(mode,
                                              predictions=predictions,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=eval_metric_ops)
        return model_fn

    @staticmethod
    def to_string(layers):
        arr = []
        for layer in layers:
            assert isinstance(layer, Layer)
            s = layer.to_string()
            arr.append(s)
        layers_str = ','.join(arr)
        return layers_str

    @staticmethod
    def parse_string(layers_str):
        arr = layers_str.split(',')
        layers = []
        for s in arr:
            layer = Layer.parse_string(s)
            layers.append(layer)
        return layers

    @staticmethod
    def load_net_from_file(filename):
        with open(filename) as fid:
            net_str = fid.readline().strip()
            net = Model.parse_string(net_str)
        return net
