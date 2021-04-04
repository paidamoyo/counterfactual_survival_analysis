import numpy as np
import os
import pandas
from utils import prepocessing

dir_path = os.path.dirname(os.path.realpath(__file__))


def generate_data(has_idx=True):
    np.random.seed(31415)
    path_factual = os.path.abspath(os.path.join(dir_path, '', 'event_pairs.csv'))
    path_covariates = os.path.abspath(os.path.join(dir_path, '', 'covariates.npy'))
    path_treatment = os.path.abspath(os.path.join(dir_path, '', 'treatment.npy'))
    print("path:{}".format(path_factual))
    data_frame = pandas.read_csv(path_factual)
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))

    # Preprocess
    x = np.load(path_covariates)
    x = np.array(x, dtype=float)
    print("missing values:", np.sum(np.isnan(x).any()))

    a = np.load(path_treatment)
    a = np.array(a, dtype=float)

    ## Factual
    y_data = data_frame[['y_f']]
    e_data = data_frame[['e_f']]

    y = np.array(y_data, dtype=float).reshape(len(y_data))
    e = np.array(e_data, dtype=float).reshape(len(e_data))

    print("x:{}, t:{}, e:{},  a:{}, len:{}".format(x[0], y[0], e[0], a, len(y)))
    assert len(x) == len(e)
    assert len(y) == len(e)
    assert len(a) == len(e)
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    if has_idx:
        train_idx = np.load(os.path.abspath(os.path.join(dir_path, '', 'train_idx.npy')))
        valid_idx = np.load(os.path.abspath(os.path.join(dir_path, '', 'valid_idx.npy')))
        test_idx = np.load(os.path.abspath(os.path.join(dir_path, '', 'test_idx.npy')))
    else:
        np.random.shuffle(idx)
        end_time = max(y)
        print("end_time:{}".format(end_time))
        print("event rate:{}".format(sum(e) / len(e)))
        print("treatment rate:{}".format(sum(a) / len(a)))
        print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], y[0], e[0], len(y)))
        num_examples = int(0.80 * len(e))
        print("num_examples:{}".format(num_examples))
        train_idx = idx[0: num_examples]
        split = int((len(y) - num_examples) / 2)

        test_idx = idx[num_examples: num_examples + split]
        valid_idx = idx[num_examples + split: len(y)]
        print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                            len(test_idx) + len(valid_idx) + num_examples))
    # print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))
    kmf_s = prepocessing.compute_km_censored(a, e, train_idx, y)
    data = 'actg175_simulated'
    columns = get_columns()

    prepocessing.formatted_data(x=x, y=y, e=e, idx=idx, a=a, name='all_idx', kmf_s=kmf_s,
                                data=data, columns=columns, dir_path=dir_path, train_idx=train_idx)
    preprocessed = {
        'train': prepocessing.formatted_data(x=x, y=y, e=e, idx=train_idx, a=a, name='train_idx', kmf_s=kmf_s,
                                             data=data, columns=columns, dir_path=dir_path, train_idx=train_idx),
        'test': prepocessing.formatted_data(x=x, y=y, e=e, idx=test_idx, a=a, name='test_idx', kmf_s=kmf_s, data=data,
                                            columns=columns, dir_path=dir_path, train_idx=train_idx),
        'valid': prepocessing.formatted_data(x=x, y=y, e=e, idx=valid_idx, a=a, name='valid_idx', kmf_s=kmf_s,
                                             data=data, columns=columns, dir_path=dir_path, train_idx=train_idx),
    }

    return preprocessed


def get_columns():
    path_data = os.path.abspath(os.path.join(dir_path, '', 'ACTG175.csv'))
    data_frame_actg = pandas.read_csv(path_data, index_col=0)
    to_drop = ['cens', 'days', 'arms', 'pidnum']
    x_data = data_frame_actg.drop(labels=to_drop, axis=1)
    print("covariate description:{}".format(x_data.describe()))
    columns = x_data.columns
    print("columns: ", columns)
    return columns
