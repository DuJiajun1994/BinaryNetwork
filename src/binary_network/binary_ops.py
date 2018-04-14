import tensorflow as tf
from tensorflow.python.framework import ops


def binary_identity(inputs, name=None):
    graph = tf.get_default_graph()
    with ops.name_scope(name, 'BinaryIdentity', [inputs]):
        with graph.gradient_override_map({"Sign": "Identity"}):
            outputs = tf.sign(inputs)
    return outputs
