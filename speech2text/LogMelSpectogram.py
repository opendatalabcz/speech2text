import tensorflow as tf
import librosa
import numpy as np


class LogMelSpectrogram(tf.keras.layers.Layer):
    def __init__(self,
                 sampling_rate,
                 n_fft,
                 frame_step,
                 lower_edge_hertz,
                 upper_edge_hertz,
                 num_mel_bins,
                 **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.frame_step = frame_step
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins = num_mel_bins

    def call(self, inputs, training=True):
        # stft(...) Computes the Short-time Fourier Transform of signals
        stft_signal = tf.signal.stft(
            inputs,
            frame_length=self.n_fft,
            frame_step=self.frame_step,
            fft_length=self.n_fft,
            pad_end=False
        )

        power_spectrograms = tf.real(stft_signal * tf.conj(stft_signal))

        # num_spectrogram_bins = power_spectrograms.shape[-1].value

        linear_to_mel_weight_matrix = tf.constant(
            np.transpose(
                librosa.filters.mel(
                    sr=self.sampling_rate,
                    n_fft=self.n_fft + 1,
                    n_mels=self.num_mel_bins,
                    fmin=self.lower_edge_hertz,
                    fmax=self.upper_edge_hertz
                )
            ),
            dtype=tf.float32
        )

        mel_spectrograms = tf.tensordot(
            power_spectrograms,
            linear_to_mel_weight_matrix,
            1
        )

        mel_spectrograms.set_shape(
            power_spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )

        return tf.log(mel_spectrograms + 1e-6)
