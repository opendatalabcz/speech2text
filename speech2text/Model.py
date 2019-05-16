import tensorflow as tf
from speech2text.SpeechNet import SpeechNet


def compute_lengths(original_lengths, params):
    """
    Computes the length of data for CTC
    """

    return tf.cast(
        tf.floor(
            (tf.cast(original_lengths, dtype=tf.float32) - params['n_fft']) /
                params['frame_step']
        ) + 1,
        tf.int32
    )


def encode_chars(labels, params):
    characters = list(params['alphabet'])

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(characters, list(range(len(characters)))), -1)
    return table.lookup(tf.string_split(labels, delimiter=''))


def decode_chars(codes, params):
    characters = list(params['alphabet'])

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(range(len(characters))), characters), '')
    return table.lookup(codes)


def decode_logits(logits, lengths, params):
    if len(tf.shape(lengths).shape) == 1:
        lengths = tf.reshape(lengths, [1])
    else:
        lengths = tf.squeeze(lengths)

    predicted_codes, _ = tf.nn.ctc_beam_search_decoder(
        tf.transpose(logits, (1, 0, 2)),
        lengths,
        merge_repeated=True
    )

    codes = tf.cast(predicted_codes[0], tf.int32)

    text = decode_chars(codes, params)

    return text, codes


def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        audio = features['audio']
        original_lengths = features['length']
    else:
        audio, original_lengths = features

    lengths = compute_lengths(original_lengths, params)

    if labels is not None:
        codes = encode_chars(labels, params)

    network = SpeechNet(params)

    is_training = mode==tf.estimator.ModeKeys.TRAIN

    logits = network(audio, training=is_training)
    text, predicted_codes = decode_logits(logits, lengths, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': logits,
            'text': tf.sparse_tensor_to_dense(
                text,
                ''
            )
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs
        )
    else:
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=codes,
                inputs=logits,
                sequence_length=lengths,
                time_major=False,
                ignore_longer_outputs_than_inputs=True
            )
        )

        mean_edit_distance = tf.reduce_mean(
            tf.edit_distance(
                tf.cast(predicted_codes, tf.int32),
                codes
            )
        )

        distance_metric = tf.metrics.mean(mean_edit_distance)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={ 'edit_distance': distance_metric }
            )

        elif mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()

            tf.summary.text(
                'train_predicted_text',
                tf.sparse_tensor_to_dense(text, '')
            )
            tf.summary.scalar('train_edit_distance', mean_edit_distance)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=global_step,
                    learning_rate=params['lr'],
                    optimizer=(params['optimizer']),
                    update_ops=update_ops,
                    clip_gradients=params['clip_gradients'],
                    summaries=[
                        "learning_rate",
                        "loss",
                        "global_gradient_norm",
                    ]
                )

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op
            )
