import tensorflow as tf
from multiprocessing import Pool
import numpy as np

def load_wave_fn(dataset_batch):


def input_fn(input_dataset, params):
    def _input_fn():
        dataset = input_dataset

        if 'max_text_length' in params and params['max_text_length'] is not None:
            print('Constraining dataset to the max_text_length')
            dataset = input_dataset[input_dataset.text.str.len() < params['max_text_length']]

        if 'min_text_length' in params and params['min_text_length'] is not None:
            print('Constraining dataset to the min_text_length')
            dataset = input_dataset[input_dataset.text.str.len() >= params['min_text_length']]

        # if 'max_wave_length' in params and params['max_wave_length'] is not None:
        #     print('Constraining dataset to the max_wave_length')

        print('Resulting dataset length: {}'.format(len(dataset)))

        def generator_fn():
            pool = Pool()
            buffer = []

            for epoch in range(params['epochs']):
                for _, row in dataset.sample(frac=1).iterrows():
                    buffer.append((row, params))

                    if len(buffer) >= params['batch_size']:

                        if params['parallelize']:
                            audios = pool.map(
                                load_wave_fn,
                                buffer
                            )
                        else:
                            audios = map(
                                load_wave_fn,
                                buffer
                            )

                        for audio, row in audios:
                            if audio is not None:
                                if np.isnan(audio).any():
                                    print('SKIPPING! NaN coming from the pipeline!')
                                else:
                                    yield (audio, len(audio)), row.text.encode()

                        buffer = []

        return tf.data.Dataset.from_generator(
            generator_fn,
            output_types=((tf.float32, tf.int32), tf.string),
            output_shapes=((None, ()), (()))
            ).padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
                (tf.TensorShape([None]), tf.TensorShape(())),
                tf.TensorShape(())
            )
        )

    return _input_fn
