import tensorflow as tf


class AtrousConvolution1D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias=True,
                 kernel_initializer=tf.glorot_normal_initializer(),
                 causal=True):
        super(AtrousConvolution1D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.causal = causal

        self.conv1d = tf.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='valid' if causal else 'same',
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

    def call(self, inputs, training=True):
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation_rate
            inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)

        return self.conv1d(inputs)
