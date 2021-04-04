import torch
import math
from utils.ot_compute import SinkhornDistance


def compute_ipm(r1, r0):
    # small epsilon is close to the true wasserstein distance
    x = torch.tensor(r0, dtype=torch.float)
    # x = r0.clone().detach().requires_grad_(True).type(torch.float)
    y = torch.tensor(r1, dtype=torch.float)
    # y = r1.clone().detach().requires_grad_(True).type(torch.float)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    dist, P, C = sinkhorn(x=x, y=y)
    # print("Sinkhorn distance: {:.3f}".format(dist.item()))
    return torch.tensor(dist, dtype=torch.double)
    # return dist.clone().detach().requires_grad_(True).type('torch.DoubleTensor')


def compute_t_reg_loss(pred_t, emp_y, a, e, c_w, t_w, args):
    # pred_t = {"mu_zero": mu_zero, "logvar_zero": logvar_zero, "mu_one": mu_one, "logvar_one": logvar_one}
    # pred_t {"logscale_zero": logscale_zero, "logeshape_zero": logeshape_zero,
    # "logscale_one": logscale_one, "logeshape_one": logeshape_one}
    constant = 1e-8
    c_w = c_w + constant  # prevent division by zero
    t_w = t_w + constant  # prevent division by zero
    if args.is_normal:
        shape = "mu"
        scale = "logvar"
    elif args.is_non_param:
        shape = "t_mu"
        scale = "c_mu"
    else:
        shape = 'logshape'
        scale = 'logscale'

    # Compute A==0
    t_0_loss = group_loss(args=args, c_w=c_w, e=e, emp_y=emp_y, pred_t=pred_t, scale=scale, shape=shape, t_w=t_w,
                          group="zero", a_select=a == 0)

    # Compute A==1
    t_1_loss = group_loss(args=args, c_w=c_w, e=e, emp_y=emp_y, pred_t=pred_t, scale=scale, shape=shape, t_w=t_w,
                          group="one", a_select=a == 1)

    size = torch.tensor(len(a)).to(dtype=torch.double)
    w_a = torch.sum(a) / size  # weight for a
    loss = (1 / w_a.to(dtype=torch.double)) * t_1_loss + (1 / (1 - w_a.to(dtype=torch.double))) * t_0_loss
    return loss / size


def group_loss(args, c_w, e, emp_y, pred_t, scale, shape, t_w, a_select, group):
    emp_y_g = emp_y[a_select]
    e_g = e[a_select]
    w_e_g = torch.sum(e_g) / torch.tensor(len(e_g)).to(dtype=torch.double)  # weight for e_g

    shape_ = (("%s_" + group) % shape)
    scale_ = (("%s_" + group) % scale)

    surv_g = neg_log_surv(y=emp_y_g[e_g == 0], shape=torch.squeeze(pred_t[shape_][e_g == 0]),
                          scale=torch.squeeze(pred_t[scale_][e_g == 0]), args=args)
    like_g = neg_log_pdf(y=emp_y_g[e_g == 1], shape=torch.squeeze(pred_t[shape_][e_g == 1]),
                         scale=torch.squeeze(pred_t[scale_][e_g == 1]), args=args)

    t_g_loss = (1 / (1 - w_e_g.to(dtype=torch.double))) * torch.sum(surv_g) + (1 / w_e_g.to(
        dtype=torch.double)) * torch.sum(like_g)

    return t_g_loss


def neg_log_surv(y, shape, scale, args):
    if args.is_non_param and args.method == 'CSA-INFO':
        pred_t = shape
        pred_c = scale
        relu = torch.nn.ReLU()
        # return torch.abs(y - pred_c) + relu(pred_c - pred_t)
        # if pred_c > pred_t (loss)
        return torch.abs(y - pred_c) + relu(pred_c - pred_t) + relu(y - pred_t)
    elif args.is_non_param:
        pred_t = shape
        relu = torch.nn.ReLU()
        # if y > pred_t  (loss) else 0 #
        return relu(y - pred_t)

    elif args.is_normal:
        logvar = scale
        stddev = torch.exp(logvar * 0.5)
        mu = shape
        constant = 1e-8
        log_t = torch.log(y + constant)
        norm_diff = (log_t - mu) / stddev.to(dtype=torch.double)
        sqrt_2 = torch.tensor(math.sqrt(2)).to(dtype=torch.double)
        cdf = 0.5 * (1.0 + torch.erf(norm_diff / sqrt_2))
        log_surv = torch.log(1 - cdf + constant)
        return -log_surv
    else:  # Weibull
        #  # lam = scale, k = shape
        log_k = shape
        log_lam = scale
        k = torch.exp(log_k)
        lam = torch.exp(log_lam)
        log_surv = - (y / lam.to(dtype=torch.double)) ** k
        return -log_surv


def neg_log_pdf(y, shape, scale, args):
    if args.is_non_param and args.method == 'CSA-INFO':
        pred_t = shape
        pred_c = scale
        relu = torch.nn.ReLU()
        # return torch.abs(y - pred_t) + relu(pred_t - pred_c)
        # if pred_t > pred_c (loss)
        # if  y > pred_c (loss)
        return torch.abs(y - pred_t) + relu(y - pred_c) + relu(pred_t - pred_c)

    elif args.is_non_param:
        pred_t = shape
        return torch.abs(y - pred_t)

    elif args.is_normal:
        logvar = scale
        mu = shape
        constant = 1e-8
        log_t = torch.log(y + constant)
        log_normal = -0.5 * (logvar + torch.pow(log_t - mu, 2) / torch.exp(logvar).to(dtype=torch.double))
        return -log_normal
    else:  # Weibull
        #  # lam = scale, k = shape
        log_k = shape
        log_lam = scale
        k = torch.exp(log_k)
        lam = torch.exp(log_lam)
        log_weibull = log_k - log_lam + (k - 1) * (torch.log(y) - log_lam) - (y / lam.to(dtype=torch.double)) ** k
        return -log_weibull


def l2_loss(scale, model):
    l2 = 0
    for param in model.parameters():
        if param.requires_grad:
            l2 += param.norm(2)
    return l2 * scale


def l1_loss(scale, model):
    l1 = 0
    for param in model.parameters():
        if param.requires_grad:
            l1 += torch.sum(torch.abs(param))
    return l1 * scale
