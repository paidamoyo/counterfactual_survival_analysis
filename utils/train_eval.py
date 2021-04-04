import torch
from lifelines.utils import concordance_index
from utils.cost import compute_t_reg_loss, compute_ipm
from utils.helpers import gVar
import pandas
import numpy as np


def train(args, model, iterator, optimizer):
    # Computes both  NLL and IPM loss of our predictions
    model.train()  # enable train model for turning on dropout, batch norm etc

    epoch_loss = 0.0
    epoch_ipm_loss = 0.0
    epoch_t_reg_loss = 0.0

    all_emp_y = torch.zeros(0)
    all_e = torch.zeros(0)
    all_a = torch.zeros(0)

    all_pred_t0 = torch.zeros(0)
    all_pred_t1 = torch.zeros(0)

    for i, batch in enumerate(iterator):
        optimizer.zero_grad()  # zero gradients computed from previous batch
        # t_reg_loss, ipm_loss, pred_t0, pred_t1
        t_reg_loss, ipm_loss, pred_t0, pred_t1 = compute_loss(batch, model, args=args)

        # print("t_reg: ", t_reg_loss, "ipm: ", ipm_loss)
        loss = t_reg_loss + args.alpha * ipm_loss
        epoch_ipm_loss += ipm_loss.item()
        epoch_t_reg_loss += t_reg_loss.item()
        epoch_loss += loss.item()  # sum loss value

        all_emp_y = torch.cat((all_emp_y, gVar(batch.y)), dim=0)
        all_e = torch.cat((all_e, gVar(batch.e)), dim=0)
        all_a = torch.cat((all_a, gVar(batch.a)), dim=0)

        all_pred_t0 = torch.cat((all_pred_t0, pred_t0), dim=0)
        all_pred_t1 = torch.cat((all_pred_t1, pred_t1), dim=0)

        loss.backward()  # compute gradients

        optimizer.step()  # update model paramters

    e_0 = all_e[all_a == 0]
    ci_index_0 = concordance_index(event_times=all_emp_y[all_a == 0].cpu().detach().numpy(),
                                   predicted_event_times=all_pred_t0.cpu().detach().numpy(),
                                   event_observed=e_0.cpu().detach().numpy())

    e_1 = all_e[all_a == 1]
    ci_index_1 = concordance_index(event_times=all_emp_y[all_a == 1].cpu().detach().numpy(),
                                   predicted_event_times=all_pred_t1.cpu().detach().numpy(),
                                   event_observed=e_1.cpu().detach().numpy())

    event_rate_0 = torch.sum(e_0) / len(e_0)
    event_rate_1 = torch.sum(e_1) / len(e_1)
    print("TRAIN: ", "ci_index 0: ", ci_index_0, "event_rate 0: ", event_rate_0, "ci_index 1: ", ci_index_1,
          "event_rate 1: ", event_rate_1)

    ci_index = (ci_index_0 + ci_index_1) * 0.5

    size = len(iterator)

    return epoch_loss / size, epoch_t_reg_loss / size, epoch_ipm_loss / size, ci_index


def evaluate(args, model, iterator):
    # Computes both  NLL and IPM loss of our predictions
    model.eval()  # enable eval model for turning off dropout, batch norm etc

    epoch_loss = 0.0
    epoch_ipm_loss = 0.0
    epoch_t_reg_loss = 0.0

    all_emp_y = torch.zeros(0)
    all_e = torch.zeros(0)
    all_a = torch.zeros(0)

    all_pred_t0 = torch.zeros(0)
    all_pred_t1 = torch.zeros(0)

    ### Ensure no gradident are calculated within the block, reduces memory consumption
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # t_reg_loss, ipm_loss, pred_t0, pred_t1
            t_reg_loss, ipm_loss, pred_t0, pred_t1 = compute_loss(batch, model, args=args)

            loss = t_reg_loss + args.alpha * ipm_loss
            epoch_ipm_loss += ipm_loss.item()
            epoch_t_reg_loss += t_reg_loss.item()
            epoch_loss += loss.item()  # sum loss value

            all_emp_y = torch.cat((all_emp_y, gVar(batch.y)), dim=0)
            all_e = torch.cat((all_e, gVar(batch.e)), dim=0)
            all_a = torch.cat((all_a, gVar(batch.a)), dim=0)

            all_pred_t0 = torch.cat((all_pred_t0, pred_t0), dim=0)
            all_pred_t1 = torch.cat((all_pred_t1, pred_t1), dim=0)

        e_0 = all_e[all_a == 0]
        e_1 = all_e[all_a == 1]

        ci_index_0 = concordance_index(event_times=all_emp_y[all_a == 0].cpu().detach().numpy(),
                                       predicted_event_times=all_pred_t0.cpu().detach().numpy(),
                                       event_observed=e_0.cpu().detach().numpy())

        ci_index_1 = concordance_index(event_times=all_emp_y[all_a == 1].cpu().detach().numpy(),
                                       predicted_event_times=all_pred_t1.cpu().detach().numpy(),
                                       event_observed=e_1.cpu().detach().numpy())

        event_rate_0 = torch.sum(e_0) / len(e_0)
        event_rate_1 = torch.sum(e_1) / len(e_1)
        print("EVALUATE: ", "ci_index 0: ", ci_index_0, "event_rate 0: ", event_rate_0, "ci_index 1: ", ci_index_1,
              "event_rate 1: ", event_rate_1)

    ci_index = (ci_index_0 + ci_index_1) * 0.5

    size = len(iterator)

    return epoch_loss / size, epoch_t_reg_loss / size, epoch_ipm_loss / size, ci_index


def compute_loss(batch, model, args):
    x = gVar(batch.x)
    y = gVar(batch.y)
    e = gVar(batch.e)
    a = gVar(batch.a)
    c_w = gVar(batch.c_w)
    t_w = gVar(batch.t_w)

    ## pred_t = {"mu_zero": mu_zero, "logvar_zero": logvar_zero, "mu_one": mu_one, "logvar_one": logvar_one}
    # pred_t {"logscale_zero": logscale_zero, "logeshape_zero": logeshape_zero,
    # "logscale_one": logscale_one, "logeshape_one": logeshape_one}
    pred_t, r0, r1 = model.forward(x=x, a=a)

    # t_reg_loss, ipm_loss, pred_t, emp_t, e
    ipm_loss = compute_ipm(r1=r1, r0=r0)
    t_reg_loss = compute_t_reg_loss(pred_t=pred_t, emp_y=y, a=a, e=e, c_w=c_w, t_w=t_w, args=args)

    if args.is_normal:
        pred_t0 = torch.squeeze(torch.exp(pred_t['mu_zero']))
        pred_t1 = torch.squeeze(torch.exp(pred_t['mu_one']))
    elif args.is_non_param:
        pred_t0 = torch.squeeze(pred_t['t_mu_zero'])
        pred_t1 = torch.squeeze(pred_t['t_mu_one'])
    else:
        # lam = scale, k = shape
        shape_0 = torch.squeeze(torch.exp(pred_t['logshape_zero']))
        scale_0 = torch.squeeze(torch.exp(pred_t['logscale_zero']))
        exp_0 = 1 / shape_0.to(dtype=torch.double)

        shape_1 = torch.squeeze(torch.exp(pred_t['logshape_one']))
        scale_1 = torch.squeeze(torch.exp(pred_t['logscale_one']))
        exp_1 = 1 / shape_1.to(dtype=torch.double)

        two = torch.tensor(2).to(dtype=torch.double)
        pred_t0 = scale_0 * (torch.log(two) ** exp_0)
        pred_t1 = scale_1 * (torch.log(two) ** exp_1)

    return t_reg_loss, ipm_loss, pred_t0, pred_t1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_params(model, x, a, args):
    with torch.no_grad():
        # pred_t = {"mu_zero": mu_zero, "logvar_zero": logvar_zero, "mu_one": mu_one, "logvar_one": logvar_one}
        # pred_t {"logscale_zero": logscale_zero, "logeshape_zero": logeshape_zero,
        # "logscale_one": logscale_one, "logeshape_one": logeshape_one}
        if args.is_normal:
            shape = "mu"
            scale = "logvar"
        else:
            shape = 'logshape'
            scale = 'logscale'

        zero_shape = '%s_zero' % shape
        zero_scale = '%s_zero' % scale

        one_shape = '%s_one' % shape
        one_scale = '%s_one' % scale
        if args.is_non_param:
            _, r0, r1 = model.forward(x=x, a=a)
            return generate_time_samples(sample_size=args.sample_size, model=model, a=a, x=x)

        else:
            pred_t, r0, r1 = model.forward(x=x, a=a)
            params_zero = {zero_shape: torch.squeeze(pred_t[zero_shape]).cpu().detach().numpy(),
                           zero_scale: torch.squeeze(pred_t[zero_scale]).cpu().detach().numpy()}

            params_one = {one_shape: torch.squeeze(pred_t[one_shape]).cpu().detach().numpy(),
                          one_scale: torch.squeeze(pred_t[one_scale]).cpu().detach().numpy()}

            t0 = pandas.DataFrame.from_dict(params_zero)
            t1 = pandas.DataFrame.from_dict(params_one)

            return {"T_0": t0, "T_1": t1, "R_0": r0.cpu().detach().numpy(),
                    "R_1": r1.cpu().detach().numpy()}


def generate_time_samples(sample_size, model, x, a):
    with torch.no_grad():
        all_pred_t0 = np.zeros((sample_size, x[a == 0].shape[0]))
        all_pred_c0 = np.zeros((sample_size, x[a == 0].shape[0]))

        all_pred_t1 = np.zeros((sample_size, x[a == 1].shape[0]))
        all_pred_c1 = np.zeros((sample_size, x[a == 1].shape[0]))
        _, r0, r1 = model.forward(x=x, a=a)
        for i in np.arange(sample_size):
            # pred_t = {"mu_zero": mu_zero, "mu_one": mu_one}
            pred_t, _, _ = model.forward(x=x, a=a)
            predicted_t0 = torch.squeeze(pred_t['t_mu_zero'])
            all_pred_t0[i] = predicted_t0.cpu().detach().numpy()

            predicted_c0 = torch.squeeze(pred_t['c_mu_zero'])
            all_pred_c0[i] = predicted_c0.cpu().detach().numpy()

            predicted_t1 = torch.squeeze(pred_t['t_mu_one'])
            all_pred_t1[i] = predicted_t1.cpu().detach().numpy()

            predicted_c1 = torch.squeeze(pred_t['c_mu_one'])
            all_pred_c1[i] = predicted_c1.cpu().detach().numpy()

        return {"T_0": np.transpose(all_pred_t0),
                "T_1": np.transpose(all_pred_t1),
                "C_0": np.transpose(all_pred_c0),
                "C_1": np.transpose(all_pred_c1),
                "R_0": r0.cpu().detach().numpy(),
                "R_1": r1.cpu().detach().numpy()}
