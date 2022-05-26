import torch
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
from models import get_model, get_loader
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def anomaly_score_histogram(y_score, y_true, anomaly_score, out_dir, f_name):
    plt.rcParams.update({'font.size': 14})
    plt.cla()
    plt.hist(y_score[y_true == 0], bins=100, density=True, color='blue', alpha=0.5, label="Normal")
    plt.hist(y_score[y_true == 1], bins=100, density=True, color='red', alpha=0.5, label="Abnormal")
    plt.xlabel(anomaly_score)
    plt.ylabel("Frequency")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.xlim(0, 1)
    plt.ylim(0, 11)
    plt.legend()
    plt.savefig('{}/{}.pdf'.format(out_dir, f_name))


def test_rec(cfgs):
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)

    module_b = []
    for state_dict in os.listdir(os.path.join(out_dir, "b")):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, "b", state_dict)))
        model.eval()
        module_b.append(model)

    print("=> Testing ... ")
    auc_l = []
    ap_l = []
    for model in module_b:
        auc, ap = test_single_model(model, test_loader, cfgs)
        auc_l.append(auc)
        ap_l.append(ap)
    print("Average results:")
    print("AUC: ", np.mean(auc_l), "±", np.std(auc_l))
    print("AP: ", np.mean(ap_l), "±", np.std(ap_l))
    print("\nResults of each model:")
    print("AUC:", auc_l)
    print("AP:", ap_l)


def test_single_model(model, test_loader, cfgs):
    model.eval()
    network = cfgs["Model"]["network"]
    with torch.no_grad():
        y_score, y_true = [], []
        for bid, (x, label, img_id) in enumerate(test_loader):
            x = x.cuda()
            if network == "AE-U":
                out, logvar = model(x)
                rec_err = (out - x) ** 2
                res = torch.exp(-logvar) * rec_err
            elif network == "AE":
                out = model(x)
                rec_err = (out - x) ** 2
                res = rec_err
            elif network == "MemAE":
                recon_res = model(x)
                rec = recon_res['output']
                res = (rec - x) ** 2

            res = res.mean(dim=(1, 2, 3))

            y_true.append(label.cpu())
            y_score.append(res.cpu().view(-1))

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap


def evaluate(cfgs):
    gpu = cfgs["Exp"]["gpu"]
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)

    module_a = []
    for state_dict in os.listdir(os.path.join(out_dir, "a")):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, "a", state_dict),
                                         map_location=torch.device('cuda:{}'.format(gpu))))
        model.eval()
        module_a.append(model)

    module_b = []
    for state_dict in os.listdir(os.path.join(out_dir, "b")):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, "b", state_dict),
                                         map_location=torch.device('cuda:{}'.format(gpu))))
        model.eval()
        module_b.append(model)

    print("=> Evaluating ... ")
    with torch.no_grad():
        y_true = []
        rec_err_l, inter_dis_l, intra_dis_l = [], [], []

        for x, label, img_id in tqdm(test_loader):
            x = x.cuda()
            if network == "AE":
                a_rec = torch.cat([model(x).squeeze(0) for model in module_a])  # N x h x w
                b_rec = torch.cat([model(x).squeeze(0) for model in module_b])
            elif network == "AE-U":
                a_rec = torch.cat([model(x)[0].squeeze(0) for model in module_a])  # N x h x w
                b_rec, unc = [], []
                for model in module_b:
                    mean, logvar = model(x)
                    b_rec.append(mean.squeeze(0))
                    unc.append(torch.exp(logvar).squeeze(0))
                b_rec = torch.cat(b_rec)
                unc = torch.cat(unc)
            elif network == "MemAE":
                a_rec = torch.cat([model(x)["output"].squeeze(0) for model in module_a])  # N x h x w
                b_rec = torch.cat([model(x)["output"].squeeze(0) for model in module_b])
            else:
                raise Exception("Invalid Network")
            mu_a = torch.mean(a_rec, dim=0)  # h x w
            mu_b = torch.mean(b_rec, dim=0)  # h x w

            # Image-Level discrepancy
            if network == "AE-U":
                var = torch.mean(unc, dim=0)

                rec_err = (x - mu_b) ** 2 / var
                inter_dis = torch.sqrt((mu_a - mu_b) ** 2 / var)
                intra_dis = torch.sqrt(torch.var(b_rec, dim=0) / var)
            else:
                rec_err = (x - mu_b) ** 2  # h x w
                inter_dis = torch.abs(mu_a - mu_b)
                intra_dis = torch.std(b_rec, dim=0)

            rec_err_l.append(rec_err.mean().cpu())
            inter_dis_l.append(inter_dis.mean().cpu())
            intra_dis_l.append(intra_dis.mean().cpu())

            y_true.append(label.cpu().item())

        rec_err_l = np.array(rec_err_l)
        inter_dis_l = np.array(inter_dis_l)
        intra_dis_l = np.array(intra_dis_l)

        y_true = np.array(y_true)

        rec_auc = metrics.roc_auc_score(y_true, rec_err_l)
        rec_ap = metrics.average_precision_score(y_true, rec_err_l)

        intra_auc = metrics.roc_auc_score(y_true, intra_dis_l)
        intra_ap = metrics.average_precision_score(y_true, intra_dis_l)

        inter_auc = metrics.roc_auc_score(y_true, inter_dis_l)
        inter_ap = metrics.average_precision_score(y_true, inter_dis_l)

        print('Rec. (ensemble)  auc:{:.3f}  ap:{:.3f}'.format(rec_auc, rec_ap))
        print('DDAD-intra       auc:{:.3f}  ap:{:.3f}'.format(intra_auc, intra_ap))
        print('DDAD-inter       auc:{:.3f}  ap:{:.3f}'.format(inter_auc, inter_ap))

        # Visualization
        intra_dis_l = (intra_dis_l - np.min(intra_dis_l)) / (np.max(intra_dis_l) - np.min(intra_dis_l))
        rec_err_l = (rec_err_l - np.min(rec_err_l)) / (np.max(rec_err_l) - np.min(rec_err_l))
        inter_dis_l = (inter_dis_l - np.min(inter_dis_l)) / (np.max(inter_dis_l) - np.min(inter_dis_l))

        anomaly_score_histogram(y_score=intra_dis_l, y_true=y_true, anomaly_score="Intra-discrepancy", out_dir=out_dir,
                                f_name="intra_hist")
        anomaly_score_histogram(y_score=inter_dis_l, y_true=y_true, anomaly_score="Inter-discrepancy", out_dir=out_dir,
                                f_name="inter_hist")
        anomaly_score_histogram(y_score=rec_err_l, y_true=y_true, anomaly_score="Reconstruction error", out_dir=out_dir,
                                f_name="rec_hist")
