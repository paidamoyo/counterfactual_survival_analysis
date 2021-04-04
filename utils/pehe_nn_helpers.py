import numpy as np


def pdist2(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    # X is an Nxp matrix (N = number of samples)
    # Y is an Mxp matrix (M = number of samples)
    # Result is an NxM distance matrix
    C = -2 * X.dot(Y.T)
    nx = np.sum(np.square(X), 1, keepdims=True)
    ny = np.sum(np.square(Y), 1, keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)


def cf_nn(x, a):
    # a is a binary array of length N indicating whether each unit received treatment or not
    # x is the Nxp matrix of covariates
    # nn_t is an array of indices of the nearest neighbors for the treatment group
    # nn_c is an array of indices of the nearest neighbors for the control group
    It = np.array(np.where(a == 1))[0, :]
    Ic = np.array(np.where(a == 0))[0, :]

    x_c = x[Ic, :]
    x_t = x[It, :]

    D = pdist2(X=x_c, Y=x_t)

    nn_t = Ic[np.argmin(D, 0)]
    nn_c = It[np.argmin(D, 1)]

    return nn_t, nn_c


def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None, f=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x, t)

    It = np.array(np.where(t == 1))[0, :]
    Ic = np.array(np.where(t == 0))[0, :]

    ycf_t = 1.0 * y[nn_t]
    eff_nn_t = ycf_t - 1.0 * y[It]
    eff_pred_t = ycf_p[It] - yf_p[It]

    # eff_pred = eff_pred_t
    # eff_nn = eff_nn_t

    ycf_c = 1.0 * y[nn_c]
    eff_nn_c = ycf_c - 1.0 * y[Ic]
    eff_pred_c = ycf_p[Ic] - yf_p[Ic]

    eff_pred = np.concatenate([eff_pred_t, eff_pred_c])  # np.vstack((eff_pred_t, eff_pred_c))
    eff_nn = np.concatenate([eff_nn_t, eff_nn_c])  # np.vstack((eff_nn_t, eff_nn_c))

    if (f is None):
        pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))
    else:
        pehe_nn = np.sqrt(f * (np.square(eff_pred - eff_nn)) / np.sum(f))

    return pehe_nn


def pdist2_mahalanobis(X, Y, cov):
    # mahalanobis distance between all pairs x in X, y in Y
    # X is an Nxp matrix (N = number of samples)
    # Y is an Mxp matrix (M = number of samples)
    # Result is an NxM distance matrix
    icov = np.linalg.inv(cov)

    num_x = X.shape[0]
    num_y = Y.shape[0]
    C = -2 * X.dot(icov).dot(Y.T)
    nx = np.diag(X.dot(icov), X.T)
    ny = np.diag(Y.dot(icov), Y.T)
    nx_repeat = np.tile(nx, (1, num_y))
    ny_repeat = np.tile(ny, (1, num_x))
    D = nx_repeat.T + ny_repeat + C
    return np.sqrt(D + 1e-8)


def cf_nn_mahalanobis(x, a):
    # a is a binary array of length N indicating whether each unit received treatment or not
    # x is the Nxp matrix o f covariates
    # nn_t is an array of indices of the nearest neighbors (per the Mahalanobis distance) for the treatment group
    # nn_c is an array of indices of the nearest neighbors (per the Mahalanobis distance) for the control group
    It = np.array(np.where(a == 1))[0, :]
    Ic = np.array(np.where(a == 0))[0, :]

    x_c = x[Ic, :]
    x_t = x[It, :]
    cov = np.cov(x)
    D = pdist2_mahalanobis(x_c, x_t, cov)
    nn_t = Ic[np.argmin(D, 0)]
    nn_c = It[np.argmin(D, 1)]

    return nn_t, nn_c


def pehe_nn_mahalanobis(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    # yf_p is an N-dimensional array of factual predictions
    # ycf_p is an N-dimensional array of counterfactual predictions
    # y,x,t are the observed outcome (N), the covariates (Nxp), the treatment (N)
    # the output is a 1-NN proxy to the PEHE
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn_mahalanobis(x, t)

    It = np.array(np.where(t == 1))[0, :]
    Ic = np.array(np.where(t == 0))[0, :]

    ycf_t = 1.0 * y[nn_t]
    eff_nn_t = ycf_t - 1.0 * y[It]
    eff_pred_t = ycf_p[It] - yf_p[It]

    # eff_pred = eff_pred_t
    # eff_nn = eff_nn_t

    ycf_c = 1.0 * y[nn_c]
    eff_nn_c = ycf_c - 1.0 * y[Ic]
    eff_pred_c = ycf_p[Ic] - yf_p[Ic]

    eff_pred = np.concatenate([eff_pred_t, eff_pred_c])  # np.vstack((eff_pred_t, eff_pred_c))
    eff_nn = np.concatenate([eff_nn_t, eff_nn_c])  # np.vstack((eff_nn_t, eff_nn_c))

    pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))

    return pehe_nn
