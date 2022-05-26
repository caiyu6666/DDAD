from models import AE, MemAE
from anomaly_data import AnomalyDetectionDataset
from torchvision import transforms
from torch.utils import data
import os


def get_model(network, mp=None, ls=None, img_size=None, mem_dim=None, shrink_thres=0.0):
    if network == "AE":
        model = AE(latent_size=ls, multiplier=mp, unc=False, img_size=img_size)
    elif network == "AE-U":
        model = AE(latent_size=ls, multiplier=mp, unc=True, img_size=img_size)
    elif network == "MemAE":
        model = MemAE(latent_size=ls, multiplier=mp, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
    else:
        raise Exception("Invalid Model Name!")

    model.cuda()
    return model


def get_loader(dataset, dtype, bs, img_size, workers=1, extra_data=0, ar=0.):
    DATA_PATH = '/usr/cy/'
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset == 'rsna':
        path = os.path.join(DATA_PATH, 'rsna-pneumonia-detection-challenge')

    elif dataset == 'vin':
        path = os.path.join(DATA_PATH, "VinCXR")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))

    dset = AnomalyDetectionDataset(main_path=path, transform=transform, mode=dtype, img_size=img_size,
                                   extra_data=extra_data, ar=ar)

    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader
