import tensorflow as tf

class HaarClassicWPD:
    def __init__(self, signal):
        self.signal = signal
        self.max_level = tf.cast(tf.math.floor(self.__log2(tf.size(signal, out_type=tf.float16))), tf.int32)
        self.tree = {'0' : tf.reshape(signal, [1, tf.size(signal)])}

    def __log2(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator

    def __high_pass_filter(self, signal):
        return tf.math.divide(
                   tf.reduce_sum(
                       tf.transpose(
                           tf.reshape(signal, [tf.shape(signal)[0]//2, 2])
                       ), 0
                   ), 2
               )

    def __low_pass_filter(self, signal):
        return tf.math.divide(
                   tf.math.subtract(
                       tf.transpose(
                           tf.reshape(signal, [tf.shape(signal)[0]//2, 2]))[0],
                       tf.transpose(
                           tf.reshape(signal, [tf.shape(signal)[0]//2, 2]))[1]
                    ), 2
                )

    def __sig_to_feature(self, signal):
        return tf.reduce_logsumexp(signal)

    def get_level(self, level):
        curr_level=1
        while curr_level <= level:
            if str(curr_level) in self.tree:
                curr_level += 1
                continue

            self.tree[str(curr_level)] = tf.map_fn(self.__low_pass_filter, self.tree[str(curr_level - 1)])
            self.tree[str(curr_level)] = tf.concat([self.tree[str(curr_level)], tf.map_fn(self.__high_pass_filter, self.tree[str(curr_level - 1)])], 0)
            curr_level += 1
        return self.tree[str(level)]

    def get_features_level(self, level):
        return tf.map_fn(self.__sig_to_feature, self.get_level(level))
