import tensorflow as tf
from binary_network import binary_layers
from model import Model
from layers import Conv, Pool, FC


class BinaryConnect(Model):
    def __init__(self):
        self._dropout_keep_prob = 0.5

    def _build_model(self, inputs, layers, num_classes, is_training):
        net = inputs
        conv_id = 0
        fc_id = 0
        for i in range(len(layers)):
            layer = layers[i]
            if isinstance(layer, Conv):
                net = binary_layers.conv2d(net,
                                           filters=layer.num_outputs,
                                           kernel_size=layer.kernel_size,
                                           padding='SAME',
                                           is_training=is_training,
                                           name='conv_{}'.format(conv_id))
                conv_id += 1
            elif isinstance(layer, Pool):
                net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='SAME')
            elif isinstance(layer, FC):
                if not isinstance(layers[i - 1], FC):
                    net = tf.contrib.layers.flatten(net)
                    # net = tf.reduce_mean(net, [1, 2])
                if i < len(layers) - 1:
                    net = binary_layers.dense(net,
                                              units=layer.num_outputs,
                                              is_final_layer=False,
                                              is_training=is_training,
                                              name='fc_{}'.format(fc_id))
                    net = tf.layers.dropout(net,
                                            rate=self._dropout_keep_prob,
                                            training=is_training)
                else:
                    net = binary_layers.dense(net,
                                              units=num_classes,
                                              is_final_layer=True,
                                              is_training=is_training,
                                              name='fc_{}'.format(fc_id))
                fc_id += 1
        return net
