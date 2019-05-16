import tensorflow as tf
from speech2text.AtrousConvolution1D import AtrousConvolution1D


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, causal, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.dilated_conv1 = AtrousConvolution1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            causal=causal
        )

        self.dilated_conv2 = AtrousConvolution1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            causal=causal
        )

        self.out = tf.layers.Conv1D(
            filters=filters,
            kernel_size=1
        )

    def call(self, inputs, training=True):
        data = tf.layers.batch_normalization(
            inputs,
            training=training
        )

        filters = self.dilated_conv1(data)
        gates = self.dilated_conv2(data)

        filters = tf.nn.tanh(filters)
        gates = tf.nn.sigmoid(gates)

        out = tf.nn.tanh(
            self.out(
                filters * gates
            )
        )

        return out + inputs, out
