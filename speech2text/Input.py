import tensorflow as tf
from multiprocessing import Pool
import numpy as np
from speech2text.DatasetManipulator import DatasetManipulator as DM
import librosa
import os


def load_wave(dataset_row_param):
    row, params = dataset_row_param

    file_path = os.path.join(DM.CUT_DATASET_PATH, row['file'])

    if params['augment']:
        print('Augmenting not yet implemented (load_wave_fn)')

    if os.path.isfile(file_path):
        return librosa.load(file_path, sr=DM.SAMPLING_RATE)[0], row
    else:
        return None, row


def input_fn(input_dataset, params, load_wave_fn=load_wave):
    def _input_fn():
        constr_dataset = input_dataset

        if 'max_text_length' in params and params['max_text_length'] is not None:
            print('Constraining dataset to the max_text_length')
            constr_dataset = input_dataset[input_dataset.label.str.len() < params['max_text_length']]

        if 'min_text_length' in params and params['min_text_length'] is not None:
            print('Constraining dataset to the min_text_length')
            constr_dataset = input_dataset[input_dataset.label.str.len() >= params['min_text_length']]

        # if 'max_wave_length' in params and params['max_wave_length'] is not None:
        #     print('Constraining dataset to the max_wave_length')

        print('Resulting dataset length: {}'.format(len(constr_dataset)))

        def generator_fn():
            pool = Pool()
            buffer = []

            for epoch in range(params['epochs']):
                if params['shuffle']:
                    dataset = constr_dataset.sample(frac=1)
                else:
                    dataset = constr_dataset

                for _, row in dataset.iterrows():
                    buffer.append((row, params))

                    if len(buffer) >= params['batch_size']:

                        if params['parallelize']:
                            audios = pool.map(load_wave_fn, buffer)
                        else:
                            audios = map(load_wave_fn, buffer)

                        for audio, a_row in audios:
                            if audio is not None:
                                # if np.isnan(audio).any():
                                #     print('SKIPPING! NaN coming from the pipeline!')
                                # else:
                                yield (audio, len(audio)), a_row.label.encode()

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
