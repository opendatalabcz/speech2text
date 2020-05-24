import tensorflow as tf

class HaarClassicWPD:
    @staticmethod
    def __log2(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator

    @staticmethod
    def __both_filters(signal):
        even_odds = tf.transpose(tf.reshape(signal, [tf.shape(signal)[0]//2, 2]))
        return tf.stack([tf.math.subtract(even_odds[0], even_odds[1]), tf.reduce_sum(even_odds, 0)], 0)/2

    @staticmethod
    def __high_pass_filter(signal):
        return tf.math.divide(
                   tf.reduce_sum(
                       tf.transpose(
                           tf.reshape(signal, [tf.shape(signal)[0]//2, 2])
                       ), 0
                   ), 2
               )

    @staticmethod
    def __low_pass_filter(signal):
        return tf.math.divide(
                   tf.math.subtract(
                       tf.transpose(
                           tf.reshape(signal, [tf.shape(signal)[0]//2, 2]))[0],
                       tf.transpose(
                           tf.reshape(signal, [tf.shape(signal)[0]//2, 2]))[1]
                    ), 2
                )

    @staticmethod
    def __sig_to_feature(signal):
        return tf.reduce_logsumexp(signal)

    @staticmethod
    def get_level(signal, level):
        signal = tf.reshape(signal, [1, tf.size(signal)])
        curr_level = 1
        while curr_level <= level:
            signal = tf.map_fn(HaarClassicWPD.__both_filters, signal)
            sig_shape = tf.shape(signal)
            signal = tf.reshape(signal, [sig_shape[0]*sig_shape[1], sig_shape[2]])
            curr_level += 1
        return signal

    @staticmethod
    def get_features_level(signal, level):
        return tf.map_fn(HaarClassicWPD.__sig_to_feature, HaarClassicWPD.get_level(signal, level))

