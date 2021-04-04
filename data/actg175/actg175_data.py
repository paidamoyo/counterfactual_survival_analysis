import numpy as np
import os
import pandas
from utils import prepocessing

dir_path = os.path.dirname(os.path.realpath(__file__))


def generate_data(has_idx=True):
    np.random.seed(31415)

    Treatment = ['arms']
    to_drop = ['cens', 'days', 'arms', 'pidnum']

    path_data = os.path.abspath(os.path.join(dir_path, '', 'ACTG175.csv'))

    data_frame = pandas.read_csv(path_data, index_col=0)
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))

    data_frame = data_frame.loc[data_frame['arms'] <= 1]
    print("selected treament 0 and 1", data_frame.shape)

    print("missing columns:", np.sum(data_frame.isna().any()))
    prepocessing.print_missing_prop(data_frame)

    # impute na
    data_frame = data_frame.fillna(data_frame.median())
    print("missing columns:", np.sum(data_frame.isna().any()))

    # Preprocess
    x_data = data_frame.drop(labels=to_drop, axis=1)
    print("covariate description:{}".format(x_data.describe()))

    columns = x_data.columns
    print("columns: ", columns)
    x = np.array(x_data, dtype=float).reshape(x_data.shape)
    print(x.shape)

    e_data = data_frame[['cens']]
    print("events description:{}".format(e_data.describe()))
    print("prop event: ", np.sum(e_data) / len(e_data))
    e = np.array(e_data, dtype=float).reshape(len(e_data))
    print(e.shape)

    y_data = data_frame[['days']]
    print("y description:{}".format(y_data.describe()))
    y = np.array(y_data, dtype=float).reshape(len(y_data))
    print(y.shape)

    a_data = data_frame[['arms']]
    print("a description:{}".format(a_data.describe()))
    a = np.array(a_data, dtype=float).reshape(len(a_data))
    print(a.shape)

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
        num_examples = int(0.70 * len(e))
        print("num_examples:{}".format(num_examples))
        train_idx = idx[0: num_examples]
        split = int((len(y) - num_examples) / 2)

        test_idx = idx[num_examples: num_examples + split]
        valid_idx = idx[num_examples + split: len(y)]
        print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                            len(test_idx) + len(valid_idx) + num_examples))
    # print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))

    kmf_s = prepocessing.compute_km_censored(a, e, train_idx, y)
    data = 'actg175'
    preprocessed = {
        'train': prepocessing.formatted_data(x=x, y=y, e=e, idx=train_idx, a=a, name='train_idx', kmf_s=kmf_s,
                                             columns=columns,
                                             train_idx=train_idx, dir_path=dir_path, data=data),
        'test': prepocessing.formatted_data(x=x, y=y, e=e, idx=test_idx, a=a, name='test_idx', kmf_s=kmf_s,
                                            columns=columns,
                                            train_idx=train_idx, dir_path=dir_path, data=data),
        'valid': prepocessing.formatted_data(x=x, y=y, e=e, idx=valid_idx, a=a, name='valid_idx', kmf_s=kmf_s,
                                             columns=columns,
                                             train_idx=train_idx, dir_path=dir_path, data=data),
    }
    return preprocessed


if __name__ == '__main__':
    generate_data()
