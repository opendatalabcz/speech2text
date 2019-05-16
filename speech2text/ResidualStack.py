import tensorflow as tf
from speech2text.ResidualBlock import ResidualBlock


class ResidualStack(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rates, causal, **kwargs):
        super(ResidualStack, self).__init__(**kwargs)

        self.blocks = [
            ResidualBlock(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                causal=causal
            )
            for dilation_rate in dilation_rates
        ]

    def call(self, inputs, training=True):
        data = inputs
        skip = 0

        for block in self.blocks:
            data, current_skip = block(data, training=training)
            skip += current_skip

        return skip
