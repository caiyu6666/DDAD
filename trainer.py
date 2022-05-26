import time
import torch
import numpy as np
import os
from tensorboardX import SummaryWriter
from test import test_single_model

from models import EntropyLossEncap, get_model, get_loader


def train_module_ab(cfgs, opt, out_dir):
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]
    extra_data = Data["extra_data"]
    ar = Data["ar"]

    Solver = cfgs["Solver"]
    bs = Solver["bs"]
    lr = Solver["lr"]
    weight_decay = Solver["weight_decay"]
    num_epoch = Solver["num_epoch"]

    if opt.mode == "a":
        train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                  extra_data=extra_data, ar=ar)
    else:  # b
        train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                  extra_data=0, ar=0)
    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)

    model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

    writer = SummaryWriter(os.path.join(out_dir, "log"))
    if network in ["AE", "AE-U"]:
        model = AE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt)
    elif network == "MemAE":
        model = MemAE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt)
    writer.close()

    model_path = os.path.join(out_dir, "{}".format(opt.mode))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i in range(5):
        model_name = os.path.join(model_path, "{}.pth".format(i))
        if not os.path.exists(model_name):
            torch.save(model.state_dict(), model_name)
            break


def AE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt):
    t0 = time.time()
    for e in range(num_epoch):
        l1s, l2s = [], []
        model.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            if cfgs["Model"]["network"] == "AE":
                out = model(x)
                rec_err = (out - x) ** 2
                loss = rec_err.mean()
                l1s.append(loss.item())
            else:  # AE-U
                mean, logvar = model(x)
                rec_err = (mean - x) ** 2
                loss1 = torch.mean(torch.exp(-logvar) * rec_err)
                loss2 = torch.mean(logvar)
                loss = loss1 + loss2
                l1s.append(rec_err.mean().item())
                l2s.append(loss2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        l2s = np.mean(l2s) if len(l2s) > 0 else 0
        writer.add_scalar('rec_err', l1s, e)
        writer.add_scalar('logvars', l2s, e)

        if e % 25 == 0:
            t = time.time() - t0
            t0 = time.time()
            if opt.mode == "b":
                auc, ap = test_single_model(model=model, test_loader=test_loader, cfgs=cfgs)
                writer.add_scalar('AUC', auc, e)
                writer.add_scalar('AP', ap, e)
                print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                      "Rec_err:{:.5f}  logvars:{:.5f}".format(opt.mode, e, num_epoch, t, auc, ap, l1s, l2s))
            else:
                print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  "
                      "Rec_err:{:.5f}  logvars:{:.5f}".format(opt.mode, e, num_epoch, t, l1s, l2s))

    return model


def MemAE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt):
    criterion_entropy = EntropyLossEncap()
    entropy_loss_weight = cfgs["Model"]["entropy_loss_weight"]
    t0 = time.time()
    for e in range(num_epoch):
        l1s = []
        model.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            out = model(x)
            rec = out['output']
            att_w = out['att']

            rec_err = (rec - x) ** 2
            loss1 = rec_err.mean()
            entropy_loss = criterion_entropy(att_w)
            loss = loss1 + entropy_loss_weight * entropy_loss

            l1s.append(rec_err.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        writer.add_scalar('rec_err', l1s, e)
        if e % 25 == 0:
            t = time.time() - t0
            t0 = time.time()

            if opt.mode == "b":
                auc, ap = test_single_model(model=model, test_loader=test_loader, cfgs=cfgs)
                writer.add_scalar('AUC', auc, e)
                writer.add_scalar('AP', ap, e)
                print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                      "Rec_err:{:.5f}".format(opt.mode, e, num_epoch, t, auc, ap, l1s))
            else:
                print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  "
                      "Rec_err:{:.5f}".format(opt.mode, e, num_epoch, t, l1s))

    return model
