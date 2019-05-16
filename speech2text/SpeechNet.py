import tensorflow as tf
from speech2text.LogMelSpectogram import LogMelSpectrogram
from speech2text.ResidualStack import ResidualStack


class SpeechNet(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(SpeechNet, self).__init__(**kwargs)

        self.to_log_mel = LogMelSpectrogram(
            sampling_rate=params['sampling_rate'],
            n_fft=params['n_fft'],
            frame_step=params['frame_step'],
            lower_edge_hertz=params['lower_edge_hertz'],
            upper_edge_hertz=params['upper_edge_hertz'],
            num_mel_bins=params['num_mel_bins']
        )

        self.expand = tf.layers.Conv1D(
            filters=params['stack_filters'],
            kernel_size=1,
            padding='same'
        )

        self.stacks = [
            ResidualStack(
                filters=params['stack_filters'],
                kernel_size=params['stack_kernel_size'],
                dilation_rates=params['stack_dilation_rates'],
                causal=params['causal_convolutions']
            )
            for _ in range(params['stacks'])
        ]

        self.out = tf.layers.Conv1D(
            filters=len(params['alphabet']) + 1,
            kernel_size=1,
            padding='same'
        )

    def call(self, inputs, training=True):
        data = self.to_log_mel(inputs)

        data = tf.layers.batch_normalization(
            data,
            training=training
        )

        if len(data.shape) == 2:
            data = tf.expand_dims(data, 0)

        data = self.expand(data)

        for stack in self.stacks:
            data = stack(data, training=training)

        data = tf.layers.batch_normalization(
            data,
            training=training
        )

        return self.out(data) + 1e-8