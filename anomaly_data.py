import os
from PIL import Image
from tqdm import tqdm
from torch.utils import data
import json


class AnomalyDetectionDataset(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", extra_data=0, ar=0.):
        super(AnomalyDetectionDataset, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        if mode == "train":
            train_normal = data_dict["train"]["0"]
            for img_name in tqdm(train_normal):
                img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                    (img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(0)
                self.img_id.append(img_name.split('.')[0])

            if extra_data > 0:
                normal_l = data_dict["train"]["unlabeled"]["0"]
                abnormal_l = data_dict["train"]["unlabeled"]["1"]

                abnormal_num = int(extra_data * ar)
                normal_num = extra_data - abnormal_num

                for img_name in tqdm(normal_l[:normal_num]):
                    img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                        (img_size, img_size), resample=Image.BILINEAR)

                    self.slices.append(img)
                    self.labels.append(0)
                    self.img_id.append(img_name.split('.')[0])

                for img_name in tqdm(abnormal_l[:abnormal_num]):
                    img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                        (img_size, img_size), resample=Image.BILINEAR)

                    self.slices.append(img)
                    self.labels.append(1)
                    self.img_id.append(img_name.split('.')[0])

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]
            for img_name in tqdm(test_normal):
                img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                    (img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(0)
                self.img_id.append(img_name.split('.')[0])

            for img_name in tqdm(test_abnormal):
                img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                    (img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(1)
                self.img_id.append(img_name.split('.')[0])

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        return img, label, img_id

    def __len__(self):
        return len(self.slices)
