import yaml
from trainer import *
from test import evaluate, test_rec
from argparse import ArgumentParser

torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config', type=str, default="config/RSNA_AE.yaml")  # config file
    parser.add_argument('--mode', dest='mode', type=str, default=None)
    # a: Module A; b: Module B; eval: evaluate; test: test reconstruction
    opt = parser.parse_args()

    with open(opt.config, "r") as f:
        cfgs = yaml.safe_load(f)

    torch.cuda.set_device(cfgs["Exp"]["gpu"])

    out_dir = cfgs["Exp"]["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if opt.mode == "a" or opt.mode == "b":
        train_module_ab(cfgs, opt, out_dir)
    elif opt.mode == "eval":
        evaluate(cfgs)
    elif opt.mode == "test":
        test_rec(cfgs)
    else:
        raise Exception("Invalid mode: {}".format(opt.mode))
