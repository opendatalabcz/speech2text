import copy
import pandas as pd
import numpy as np
import tensorflow as tf
from speech2text.Model import model_fn
from speech2text.Input import input_fn


def dataset_params(batch_size=32,
                   epochs=50000,
                   parallelize=True,
                   max_text_length=15,
                   min_text_length=None,
                   max_wave_length=80000,
                   shuffle=True,
                   random_shift_min=-4000,
                   random_shift_max=4000,
                   random_stretch_min=0.7,
                   random_stretch_max=1.3,
                   random_noise=0.75,
                   random_noise_factor_min=0.2,
                   random_noise_factor_max=0.5,
                   augment=False):
    return {
        'parallelize': parallelize,
        'shuffle': shuffle,
        'max_text_length': max_text_length,
        'min_text_length': min_text_length,
        'max_wave_length': max_wave_length,
        'random_shift_min': random_shift_min,
        'random_shift_max': random_shift_max,
        'random_stretch_min': random_stretch_min,
        'random_stretch_max': random_stretch_max,
        'random_noise': random_noise,
        'random_noise_factor_min': random_noise_factor_min,
        'random_noise_factor_max': random_noise_factor_max,
        'epochs': epochs,
        'batch_size': batch_size,
        'augment': augment
    }


def experiment_params(data,
                      optimizer='Adam',
                      lr=1e-4,
                      alphabet=' aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž',
                      causal_convolutions=True,
                      stack_dilation_rates=[1, 3, 9, 27, 81],
                      stacks=2,
                      stack_kernel_size=3,
                      stack_filters=32,
                      sampling_rate=16000,
                      n_fft=160*4,
                      frame_step=160,
                      lower_edge_hertz=0,
                      upper_edge_hertz=8000,
                      num_mel_bins=160,
                      clip_gradients=None,
                      codename='regular',
                      **kwargs):
    params = {
        'optimizer': optimizer,
        'lr': lr,
        'data': data,
        'alphabet': alphabet,
        'causal_convolutions': causal_convolutions,
        'stack_dilation_rates': stack_dilation_rates,
        'stacks': stacks,
        'stack_kernel_size': stack_kernel_size,
        'stack_filters': stack_filters,
        'sampling_rate': sampling_rate,
        'n_fft': n_fft,
        'frame_step': frame_step,
        'lower_edge_hertz': lower_edge_hertz,
        'upper_edge_hertz': upper_edge_hertz,
        'num_mel_bins': num_mel_bins,
        'clip_gradients': clip_gradients,
        'codename': codename
    }

    if kwargs is not None and 'data' in kwargs:
        params['data'] = { **params['data'], **kwargs['data'] }
        del kwargs['data']

    if kwargs is not None:
        params = { **params, **kwargs }

    return params


def experiment_name(params, excluded_keys=['alphabet', 'data', 'lr', 'clip_gradients']):

    def represent(key, value):
        if key in excluded_keys:
            return None
        else:
            if isinstance(value, list):
                return '{}_{}'.format(key, '_'.join([str(v) for v in value]))
            else:
                return '{}_{}'.format(key, value)

    parts = filter(
        lambda p: p is not None,
        [
            represent(k, params[k])
            for k in sorted(params.keys())
        ]
    )

    return '/'.join(parts)


def experiment(data_params=dataset_params(), **kwargs):
    params = experiment_params(data_params, **kwargs)

    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='stats/{}'.format(experiment_name(params)),
        params=params
    )

    data = pd.read_csv('datasets/cpm_cut/data.csv')
    msk = np.random.rand(len(data)) < 0.8
    train_data = data[msk]
    eval_data = data[~msk]

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn(
            train_data,
            params['data']
        )
    )

    features = {
        "audio": tf.placeholder(dtype=tf.float32, shape=[None]),
        "length": tf.placeholder(dtype=tf.int32, shape=[])
    }

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features
    )

    best_exporter = tf.estimator.BestExporter(
      name="best_exporter",
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=5
    )

    eval_params = copy.deepcopy(params['data'])
    eval_params['augment'] = False

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn(
            eval_data,
            eval_params
        ),
        throttle_secs=60*30,
        exporters=best_exporter
    )

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )
