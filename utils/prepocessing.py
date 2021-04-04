import numpy as np
from lifelines import KaplanMeierFitter
from utils import pehe_nn_helpers
import pandas
import os


def compute_km_censored(a, e, train_idx, y):
    a_train = a[train_idx]
    y_train = y[train_idx]
    e_train = e[train_idx]

    # Censored Survival
    c_kmf_0 = estimate_km(y=y_train[a_train == 0], e=1 - e_train[a_train == 0])
    c_kmf_1 = estimate_km(y=y_train[a_train == 1], e=1 - e_train[a_train == 1])

    # NonCensored Survival
    t_kmf_0 = estimate_km(y=y_train[a_train == 0], e=e_train[a_train == 0])
    t_kmf_1 = estimate_km(y=y_train[a_train == 1], e=e_train[a_train == 1])

    return {'c_kmf_0': c_kmf_0, 'c_kmf_1': c_kmf_1, "t_kmf_0": t_kmf_0, "t_kmf_1": t_kmf_1}


def estimate_km(y, e):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=y, event_observed=e)
    return kmf


def formatted_data(x, y, e, idx, a, name, kmf_s, columns, train_idx, dir_path, data):
    death_time = np.array(y[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])
    treatment = np.array(a[idx])

    train_x = x[train_idx]
    cov = np.cov(train_x.T)

    train_a = a[train_idx]
    train_y = y[train_idx]
    train_e = e[train_idx]

    train_e_0, train_x_0, train_y_0 = get_train_a(train_a == 0, train_e, train_x, train_y)
    train_e_1, train_x_1, train_y_1 = get_train_a(train_a == 1, train_e, train_x, train_y)

    c_w_0 = np.array(kmf_s['c_kmf_0'].predict(death_time))
    c_w_1 = np.array(kmf_s['c_kmf_1'].predict(death_time))
    print("c_w_0: ", c_w_0.shape, "c_w_1: ", c_w_1.shape)

    t_w_0 = np.array(kmf_s['t_kmf_0'].predict(death_time))
    t_w_1 = np.array(kmf_s['t_kmf_1'].predict(death_time))
    print("t_w_0: ", t_w_0.shape, "t_w_1: ", t_w_1.shape)

    c_w = [treatment[i] * c_w_1[i] + (1 - treatment[i]) * c_w_0[i] for i in np.arange(treatment.shape[0])]
    c_w = np.array(c_w)
    print("c_w: ", c_w.shape)

    t_w = [treatment[i] * t_w_1[i] + (1 - treatment[i]) * t_w_0[i] for i in np.arange(treatment.shape[0])]
    t_w = np.array(t_w)
    print("t_w: ", c_w.shape)

    print(name, " event rate fold:{}".format(sum(e[idx]) / len(e[idx])))
    print(name, " treatment rate fold:{}".format(sum(a[idx]) / len(a[idx])))
    path_idx = os.path.abspath(os.path.join(dir_path, '', name))
    np.save(path_idx, idx)

    ## nn_cf_y nn_cf_e
    nn_cf_y = np.zeros(len(treatment))
    nn_cf_e = np.zeros(len(treatment))

    nn_c = compute_nn(treatment == 1, train_x_a=train_x_0, covariates=covariates, cov=cov)
    nn_cf_y[treatment == 1] = train_y_0[nn_c]
    nn_cf_e[treatment == 1] = train_e_0[nn_c]

    nn_t = compute_nn(treatment == 0, train_x_a=train_x_1, covariates=covariates, cov=cov)
    nn_cf_y[treatment == 0] = train_y_1[nn_t]
    nn_cf_e[treatment == 0] = train_e_1[nn_t]

    survival_data = {'x': covariates, 'y': death_time, 'e': censoring, 'a': treatment, 'c_w': c_w, 't_w': t_w,
                     'nn_cf_y': nn_cf_y, 'nn_cf_e': nn_cf_e}

    concatenate_save(e=censoring, x=covariates, name=name, y=death_time, a=treatment, columns=columns, nn_cf_y=nn_cf_y,
                     nn_cf_e=nn_cf_e, dir_path=dir_path, data=data)
    return survival_data


def get_train_a(a_selector, train_e, train_x, train_y):
    train_x_0 = train_x[a_selector]
    train_y_0 = train_y[a_selector]
    train_e_0 = train_e[a_selector]
    return train_e_0, train_x_0, train_y_0


def compute_nn(a_select, covariates, train_x_a, cov):
    print("covariates: ", covariates.shape, "cov: ", cov.shape, "train_x_a: ", train_x_a.shape)
    # dist_c = pehe_nn_helpers.pdist2_mahalanobis(X=covariates[a_select], Y=train_x_a,
    #                                             cov=cov)

    dist_c = pehe_nn_helpers.pdist2(X=covariates[a_select], Y=train_x_a)
    nn = np.argmin(dist_c, axis=1)
    assert (len(nn) == np.sum(a_select))
    print("nn: ", len(nn))
    return nn


def concatenate_save(e, x, y, name, a, columns, nn_cf_e, nn_cf_y, data, dir_path):
    all_columns = np.concatenate((columns, np.array(['time', 'event', 'nn_cf_e', 'nn_cf_y'])), axis=0)
    print(all_columns, all_columns.shape)
    reshaped_censoring = np.expand_dims(e, axis=1)
    reshaped_death_time = np.expand_dims(y, axis=1)
    reshaped_death_cf_time = np.expand_dims(nn_cf_y, axis=1)
    reshaped_death_cf_censoring = np.expand_dims(nn_cf_e, axis=1)
    reshaped_a = np.expand_dims(a, axis=1)
    all_data = np.concatenate(
        (x, reshaped_death_time, reshaped_censoring, reshaped_death_cf_censoring, reshaped_death_cf_time), axis=1)
    print(name, all_data.shape)
    print(name + ' a==0: ', all_data[a == 0].shape)
    print(name + ' a==1: ', all_data[a == 1].shape)
    save(all_columns=all_columns, all_data=all_data[a == 0], name=name + '_a0', data=data, dir_path=dir_path)
    save(all_columns=all_columns, all_data=all_data[a == 1], name=name + '_a1', data=data, dir_path=dir_path)
    save(all_columns=np.concatenate((all_columns, np.array(['treatment'])), axis=0),
         all_data=np.concatenate((all_data, reshaped_a), axis=1), name=name, data=data, dir_path=dir_path)


def save(all_columns, all_data, name, data, dir_path):
    # path = os.path.abspath(os.path.join(dir_path, '', 'framingham_' + name + '.csv'))
    path = os.path.abspath(os.path.join(dir_path, '', data + '_' + name + '.csv'))
    sub_data = pandas.DataFrame(all_data, columns=all_columns)
    return sub_data.to_csv(path, encoding='utf-8', index=False)


def print_missing_prop(covariates):
    missing = np.array(np.isnan(covariates), dtype=float)
    shape = np.shape(covariates)
    proportion = np.sum(missing) / (shape[0] * shape[1])
    print("missing_proportion:{}".format(proportion))
