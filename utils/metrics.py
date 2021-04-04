import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter

seed = 31415
np.random.seed(seed)

fontsize = 18
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

plt.rc('font', **font)
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.

sns.set_style('white')
sns.set_context('paper')
sns.set()
title_fontsize = 12
label_fontsize = 10


def plot_cost(training, validation, name, model, epochs, best_epoch, args):
    x = np.arange(start=0, stop=len(training), step=1).tolist()
    constant = 1e-10
    plt.figure()
    plt.xlim(min(x), max(x))
    plt.ylim(min(min(training), min(validation), 0) - constant, max(max(training), max(validation)) + constant)
    plt.plot(x, training, color='blue', linestyle='-', label='training')
    plt.plot(x, validation, color='green', linestyle='-', label='validation')
    plt.axvline(x=best_epoch, color='red')
    title = 'Training {} {}: epochs={}, best epoch={} '.format(model, name, epochs, best_epoch)
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.legend(loc='best', fontsize=10)
    plt.savefig('plots/run_{}_alpha_{}/{}_{}'.format(args.config_num, args.alpha, model, name))
