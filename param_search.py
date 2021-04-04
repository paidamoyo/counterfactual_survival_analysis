import os
import numpy as np
from subprocess import call
import argparse


# Adapted from: https://github.com/clinicalml/cfrnet/tree/master/cfr

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file, 'r') as f:
        for l in f:
            l = l.strip()
            if len(l) > 0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs) > 0:
                    k, v = (vs[0], eval(vs[1]))
                    if not isinstance(v, list):
                        v = [v]
                    cfg[k] = v
    return cfg


def sample_config(configs, num):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts), 1)[0]
        cfg_sample[k] = opts[c]
    cfg_sample['config_num'] = num
    return cfg_sample


def cfg_string(cfg):
    ks = sorted(cfg.keys())
    # cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    cfg_str = ''
    for k in ks:
        if k == 'config_num':
            continue
        elif cfg_str != '':
            cfg_str = cfg_str + ', ' + ('%s:%s' % (k, str(cfg[k])))
        else:
            cfg_str = ('%s:%s' % (k, str(cfg[k])))
    return cfg_str.lower()


def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    used_cfgs = read_used_cfgs(used_cfg_file)
    return cfg_str in used_cfgs


def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())

    return used_cfgs


def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)


def run(cfg_file, num_runs):
    configs = load_config(cfg_file)

    outdir = 'config'
    used_cfg_file = '%s/used_configs.txt' % outdir

    # if not os.path.isfile(used_cfg_file):
    f = open(used_cfg_file, 'w')
    f.close()

    for i in range(num_runs):
        cfg = sample_config(configs, num=i + 1)
        if is_used_cfg(cfg, used_cfg_file):
            print('Configuration used, skipping')
            continue
        save_used_cfg(cfg, used_cfg_file)

        print('------------------------------')
        print('Run %d of %d:' % (i + 1, num_runs))
        print('------------------------------')
        print('\n'.join(
            ['%s: %s' % (str(k), str(v)) for k, v in cfg.items() if k in configs.keys() and len(configs[k]) > 1]))

        flags = ' '.join('--%s %s' % (k, str(v)) for k, v in cfg.items())
        call('python train.py %s' % flags, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Search')
    # param search arguments
    # <config_file> <num_runs>
    parser.add_argument('--config_file', type=str, default='config/configs.txt', help='path to configuration file')
    parser.add_argument('--num_runs', type=int, default=30, help='number of generated models')

    args = parser.parse_args()
    run(cfg_file=args.config_file, num_runs=args.num_runs)
