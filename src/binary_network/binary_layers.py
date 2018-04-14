import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from binary_network.binary_ops import binary_identity
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers.python.layers import initializers

weight_decay = 0.0005


@add_arg_scope
def conv2d(inputs,
           filters,
           kernel_size,
           padding,
           is_training,
           name):
    with tf.variable_scope(name):
        if type(kernel_size) == int:
            kernel_size = [kernel_size, kernel_size]
        num_inputs = inputs.get_shape().as_list()[3]
        w = tf.get_variable('weights', [kernel_size[0], kernel_size[1], num_inputs, filters],
                            initializer=initializers.xavier_initializer(),
                            regularizer=layers.l2_regularizer(scale=weight_decay))
        w = binary_identity(w)
        outputs = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding=padding)
        outputs = layers.batch_norm(outputs,
                                    is_training=is_training,
                                    decay=0.9,
                                    center=True,
                                    scale=True,
                                    updates_collections=None)
        outputs = tf.nn.relu(outputs)
    return outputs


@add_arg_scope
def dense(inputs,
          units,
          is_final_layer,
          is_training,
          name):
    with tf.variable_scope(name):
        num_inputs = inputs.get_shape().as_list()[1]
        w = tf.get_variable('weights', [num_inputs, units],
                            initializer=initializers.xavier_initializer(),
                            regularizer=layers.l2_regularizer(scale=weight_decay))
        w = binary_identity(w)
        outputs = tf.matmul(inputs, w)
        if is_final_layer:
            b = tf.get_variable('biases', [units],
                                initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, b)
        else:
            outputs = layers.batch_norm(outputs,
                                        is_training=is_training,
                                        decay=0.9,
                                        center=True,
                                        scale=True,
                                        updates_collections=None)
            outputs = tf.nn.relu(outputs)
    return outputs
