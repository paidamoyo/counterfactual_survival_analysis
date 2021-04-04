import os
import argparse
import random
import torch
import torch.optim as optim
import logging
import time
from data.actg175_simulated import actg75_simulated_data
from data.actg175 import actg175_data

import numpy as np

from model.csa import CSA
from model.encoder import Encoder
from model.decoder_normal import DecoderNormal
from model.decoder_weibull import DecoderWeibull
from model.decoder_non_param import DecoderNonParam
from utils import helpers, train_eval
from data.custom_batch import build_iterator
from utils.metrics import plot_cost
from utils.cost import l1_loss, l2_loss


def init_config():
    parser = argparse.ArgumentParser(description='Causal Survival Analysis')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, default='simulated', help='dataset in [simulated]')
    parser.add_argument('--GPUID', type=str, default='0', help='GPU ID')
    parser.add_argument('--config_num', type=int, help='use config line number')
    parser.add_argument('--alpha', type=float, help='IPM weight')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--l1_reg', type=float)
    parser.add_argument('--l2_reg', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--require_improvement', type=int)
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', type=str, help='select from {SR, CSA, CSA-INFO, AFT, AFT-Weibull}')

    args = parser.parse_args()

    if not os.path.isdir('./matrix'):
        os.makedirs('./matrix')
    if not os.path.isdir('./plots'):
        os.makedirs('./plots')
    if not os.path.isdir('./matrix/run_{}_alpha_{}'.format(args.config_num, args.alpha)):
        os.makedirs('./matrix/run_{}_alpha_{}'.format(args.config_num, args.alpha))
    if not os.path.isdir('./plots/run_{}_alpha_{}'.format(args.config_num, args.alpha)):
        os.makedirs('./plots/run_{}_alpha_{}'.format(args.config_num, args.alpha))
    if not os.path.isdir('./logs'):
        os.makedirs('./logs')
    if not os.path.isdir('./results'):
        os.makedirs('./results')

    args.is_non_param = True
    args.is_stochastic = True
    args.is_normal = False  # 0: AFT=Weibull, 1: AFT=log-normal?

    if args.method == 'SR':
        args.is_stochastic = False

    if 'AFT' in args.method:
        args.is_non_param = False

    if args.method == 'AFT':
        args.is_normal = True

    return args


def save_results(a, name, x, fold):
    results = train_eval.save_params(model=model, x=x, a=a, args=args)

    t0 = results["T_0"]
    t1 = results["T_1"]

    r0 = results['R_0']
    r1 = results['R_1']
    print(name, " R_0:", r0.shape, "R_1:", r1.shape)

    if args.is_non_param:
        c1 = results["C_1"]
        c0 = results["C_0"]

        np.save('matrix/run_{}_alpha_{}/{}_pred_t0_{}'.format(args.config_num, args.alpha, fold, name), t0)
        np.save('matrix/run_{}_alpha_{}/{}_pred_t1_{}'.format(args.config_num, args.alpha, fold, name), t1)

        if fold == 'Test':
            np.save('matrix/run_{}_alpha_{}/{}_pred_c0_{}'.format(args.config_num, args.alpha, fold, name), c0)
            np.save('matrix/run_{}_alpha_{}/{}_pred_c1_{}'.format(args.config_num, args.alpha, fold, name), c1)
    else:
        t0.to_csv('matrix/run_{}_alpha_{}/{}_pred_t0_{}.csv'.format(args.config_num, args.alpha, fold, name),
                  index=False)
        t1.to_csv('matrix/run_{}_alpha_{}/{}_pred_t1_{}.csv'.format(args.config_num, args.alpha, fold, name),
                  index=False)
    if fold == 'Test':
        np.save('matrix/run_{}_alpha_{}/{}_r0_{}'.format(args.config_num, args.alpha, fold, name), r0)
        np.save('matrix/run_{}_alpha_{}/{}_r1_{}'.format(args.config_num, args.alpha, fold, name), r1)


def plot_metrics():
    algorithm = 'CSA'
    plot_cost(training=all_train_loss, validation=all_valid_loss, model=algorithm, name="Cost",
              epochs=data['epochs'],
              best_epoch=best_epoch, args=args)
    plot_cost(training=all_train_ci, validation=all_valid_ci, model=algorithm, name="CI",
              epochs=data['epochs'],
              best_epoch=best_epoch, args=args)
    plot_cost(training=all_train_ipm, validation=all_valid_ipm, model=algorithm, name="IPM_loss",
              epochs=data['epochs'], best_epoch=best_epoch, args=args)
    plot_cost(training=all_train_t_reg, validation=all_valid_t_reg, model=algorithm, name="T_reg_loss",
              epochs=data['epochs'], best_epoch=best_epoch, args=args)


if __name__ == '__main__':
    args = init_config()
    GPUID = args.GPUID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
    ### Logging
    log_file = 'logs/model_{}_alpha_{}.log'.format(args.config_num, args.alpha)
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)

    actg75_simulated = {"preprocess": actg75_simulated_data, "epochs": 300}
    actg175 = {"preprocess": actg175_data, "epochs": 300}

    all_datasets = {"actg": actg175, "actg_simulated": actg75_simulated}
    data = all_datasets[args.dataset]

    data_set = data['preprocess'].generate_data()

    model = CSA

    ### Load DATA
    train_data, valid_data, test_data = data_set['train'], data_set['valid'], data_set['test']

    ### Set random seed for determinstic result
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed=args.seed)

    MODEL_NAME = 'results/CSA_{}.pt'.format(args.config_num)

    ### Torch device for putting tensors into GPU if available
    cuda_device = torch.device('cuda')
    cpu_device = torch.device('cpu')
    cuda_tensor = 'torch.cuda.DoubleTensor'
    cpu_tensor = 'torch.DoubleTensor'
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = cuda_device
        torch.set_default_tensor_type(cuda_tensor)
        torch.cuda.manual_seed(seed=args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_default_tensor_type(cpu_tensor)
        device = cpu_device

    model_init = helpers.uniform_initializer(0.01)
    enc = Encoder(input_dim=train_data['x'].shape[1], hidden_dim=args.hidden_dim,
                  dropout=args.dropout, model_init=model_init)

    args.n_components = 0
    dec = None
    if args.method == 'SR':
        dec = DecoderNonParam(output_dim=1, hidden_dim=args.hidden_dim,
                              dropout=args.dropout, model_init=model_init, is_stochastic=args.is_stochastic)
    elif 'CSA' in args.method:
        dec = DecoderNonParam(output_dim=1, hidden_dim=args.hidden_dim,
                              dropout=args.dropout, model_init=model_init, is_stochastic=args.is_stochastic)
    elif args.method == 'AFT':
        dec = DecoderNormal(output_dim=1, hidden_dim=args.hidden_dim,
                            dropout=args.dropout, model_init=model_init)
    elif 'AFT-Weibull' in args.method:
        dec = DecoderWeibull(output_dim=1, hidden_dim=args.hidden_dim,
                             dropout=args.dropout, model_init=model_init)

    else:
        print("choose method")
        exit(1)

    print(args)
    logging.debug(args)

    model = CSA(encoder=enc, decoder=dec).to(device=device)
    print(model)
    logging.debug(model)
    parameters = helpers.count_parameters(model)
    # assert (parameters == N)
    print_param = "The model has trainable parameters:{}".format(parameters)
    print(print_param)
    logging.debug(print_param)

    ### Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    ### Build a batch iterator for x, y, e, a,c_w
    iterators = build_iterator(args=args, train_data=train_data, valid_data=valid_data, test_data=test_data)

    ### Finally train model
    best_ipm = float('inf')
    best_ci = 0.0
    best_epoch = 0
    all_train_loss, all_valid_loss = [], []
    all_train_ci, all_valid_ci = [], []
    all_train_t_reg, all_valid_t_reg = [], []
    all_train_ipm, all_valid_ipm = [], []

    for epoch in range(data['epochs']):
        start_time = time.time()

        train_loss, train_t_reg, train_ipm, train_ci = train_eval.train(model=model,
                                                                        iterator=iterators["train_iterator"],
                                                                        optimizer=optimizer, args=args)
        # print(torch.cuda.memory_snapshot())
        if use_cuda:
            print()
            print("MEM USAGE: ", torch.cuda.memory_allocated(device=0))
            print()

        valid_loss, valid_t_reg, valid_ipm, valid_ci = train_eval.evaluate(model=model,
                                                                           iterator=iterators[
                                                                               "valid_iterator"], args=args)

        all_train_ipm.append(train_ipm)
        all_valid_ipm.append(valid_ipm)

        all_train_t_reg.append(train_t_reg)
        all_valid_t_reg.append(valid_t_reg)

        all_train_loss.append(train_loss)
        all_valid_loss.append(valid_loss)

        all_train_ci.append(train_ci)
        all_valid_ci.append(valid_ci)

        end_time = time.time()

        epoch_mins, epoch_sec = train_eval.epoch_time(start_time=start_time, end_time=end_time)

        print("valid_ipm: ", valid_ipm, "best_ipm: ", best_ipm, "valid_ci: ", valid_ci, "best_ci: ",
              best_ci)
        improved_str = ''
        ipm_warm_start = data['epochs'] * 0.2
        # if args.alpha > 0 and valid_ipm <= best_ipm and valid_ci >= best_ci and epoch > ipm_warm_start:
        if args.alpha > 0 and valid_ipm <= best_ipm and epoch > ipm_warm_start:
            best_ipm = valid_ipm
            best_epoch = epoch
            best_ci = valid_ci
            torch.save(model.state_dict(), MODEL_NAME)
            improved_str = '*'

        elif args.alpha == 0 and valid_ci >= best_ci:
            best_epoch = epoch
            best_ci = valid_ci
            torch.save(model.state_dict(), MODEL_NAME)
            improved_str = '*'

        print_epoch = "Epoch:{} | Time: {}m {}s".format(epoch + 1, epoch_mins, epoch_sec)
        print(print_epoch)
        logging.debug(print_epoch)

        print_train = "\t Train Loss:{} |  t_reg:{} |  ipm:{} |ci:{}".format(train_loss, train_t_reg, train_ipm,
                                                                             train_ci)
        print(print_train)
        logging.debug(print_train)
        print_val = "\t Val Loss: {} | t_reg: {}| ipm:{} | ci:{} | {}".format(valid_loss, valid_t_reg,
                                                                              valid_ipm, valid_ci, improved_str)
        print(print_val)
        logging.debug(print_val)
        print_reg = "Regularization l1 loss:{} | l2 loss:{}".format(l1_loss(scale=args.l1_reg, model=model),
                                                                    l2_loss(scale=args.l2_reg, model=model))
        print(print_reg)
        logging.debug(print_reg)

        if args.n_components > 0:
            var_one = np.exp(dec.one_log_var_set.cpu().detach().numpy())
            print_var_one = "\t Component Variances One: {}".format(var_one)
            print(print_var_one)
            logging.debug(print_var_one)

            var_zero = np.exp(dec.zero_log_var_set.cpu().detach().numpy())
            print_var_zero = "\t Component Variances Zero: {}".format(var_zero)
            logging.debug(print_var_zero)

    print("\t Training Complete, Loading Saved Model !!!")
    logging.debug("\t Training Complete, Loading Saved Model !!!")

    model.load_state_dict(torch.load(MODEL_NAME))
    train_loss, train_t_reg, train_ipm, train_ci = train_eval.evaluate(model=model,
                                                                       iterator=iterators["train_iterator"],
                                                                       args=args)

    print_train = "\t Train Loss:{} |  t_reg:{} | ipm:{} | ci:{}".format(train_loss, train_t_reg, train_ipm, train_ci)
    print(print_train)
    logging.debug(print_train)

    # np.mean(epoch_loss), np.mean(epoch_t_reg_loss), np.mean(epoch_ipm_loss), np.mean(epoch_ci_index)
    valid_loss, valid_t_reg, valid_ipm, valid_ci = train_eval.evaluate(model=model,
                                                                       iterator=iterators["valid_iterator"],
                                                                       args=args)
    print_val = "\t Val Loss: {} | t_reg: {}| ipm:{} | ci:{}".format(valid_loss, valid_t_reg, valid_ipm, valid_ci)
    print(print_val)
    logging.debug(print_val)

    test_loss, test_t_reg, test_ipm, test_ci = train_eval.evaluate(model=model,
                                                                   iterator=iterators["test_iterator"],
                                                                   args=args)
    print_test = "\t Test Loss: {} | t_reg: {}| ipm:{} | ci:{}".format(test_loss, test_t_reg, test_ipm, test_ci)
    print(print_test)
    logging.debug(print_test)

    save_results(a=torch.Tensor(test_data['a']), name='F', x=torch.Tensor(test_data['x']), fold='Test')
    save_results(a=1 - torch.Tensor(test_data['a']), name='CF', x=torch.Tensor(test_data['x']), fold='Test')

    save_results(a=torch.Tensor(valid_data['a']), name='F', x=torch.Tensor(valid_data['x']), fold='Valid')
    save_results(a=1 - torch.Tensor(valid_data['a']), name='CF', x=torch.Tensor(valid_data['x']), fold='Valid')

    if args.n_components > 0:
        np.save('matrix/run_{}_alpha_{}/{}_one_log_var'.format(args.config_num, args.alpha, 'Test'),
                dec.one_log_var_set.cpu().detach().numpy())
        np.save('matrix/run_{}_alpha_{}/{}_zero_log_var'.format(args.config_num, args.alpha, 'Test'),
                dec.zero_log_var_set.cpu().detach().numpy())
    plot_metrics()
