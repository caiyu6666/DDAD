import os
import time

from PIL import Image
from tqdm import tqdm
from torch.utils import data
import json
from joblib import Parallel, delayed


def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


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

        print("Loading images")
        if mode == "train":
            # t0 = time.time()
            train_normal = data_dict["train"]["0"]
            # self.slices += Parallel(n_jobs=-1, verbose=1)(delayed(
            #     lambda file: Image.open(os.path.join(self.root, "train_png_512", file)).convert("L").resize(
            #         (img_size, img_size), resample=Image.BILINEAR))(file) for file in train_normal)
            # self.labels += len(train_normal) * [0]
            # self.img_id += [img_name.split('.')[0] for img_name in train_normal]
            # print("Loaded {} normal images. {:.3f}s".format(len(train_normal), time.time()-t0))
            # for img_name in tqdm(train_normal):
            #     img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
            #         (img_size, img_size), resample=Image.BILINEAR)
            #     self.slices.append(img)
            #     self.labels.append(0)
            #     self.img_id.append(img_name.split('.')[0])

            normal_l = data_dict["train"]["unlabeled"]["0"]
            abnormal_l = data_dict["train"]["unlabeled"]["1"]
            if extra_data > 0:
                abnormal_num = int(extra_data * ar)
                normal_num = extra_data - abnormal_num

                # for img_name in tqdm(normal_l[:normal_num]):
                #     img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                #         (img_size, img_size), resample=Image.BILINEAR)
                #
                #     self.slices.append(img)
                #     self.labels.append(0)
                #     self.img_id.append(img_name.split('.')[0])

                # for img_name in tqdm(abnormal_l[:abnormal_num]):
                #     img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
                #         (img_size, img_size), resample=Image.BILINEAR)
                #
                #     self.slices.append(img)
                #     self.labels.append(1)
                #     self.img_id.append(img_name.split('.')[0])
            else:
                abnormal_num = 0
                normal_num = 0

            train_l = train_normal + normal_l[:normal_num] + abnormal_l[:abnormal_num]
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "train_png_512"), train_l, img_size)
            self.labels += (len(train_normal) + normal_num) * [0] + abnormal_num * [1]
            self.img_id += [img_name.split('.')[0] for img_name in train_l]
            print("Loaded {} normal images, "
                  "{} (unlabeled) normal images, "
                  "{} (unlabeled) abnormal images. {:.3f}s".format(len(train_normal), normal_num, abnormal_num,
                                                                   time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "train_png_512"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

            # test normal
            # t0 = time.time()
            # self.slices += Parallel(n_jobs=10)(delayed(
            #     lambda file: Image.open(os.path.join(self.root, "train_png_512", file)).convert("L").resize(
            #         (img_size, img_size), resample=Image.BILINEAR))(file) for file in test_normal)
            # self.labels += len(test_normal) * [0]
            # self.img_id += [img_name.split('.')[0] for img_name in test_normal]
            # print("Loaded {} test normal images. {:.3f}s".format(len(test_normal), time.time() - t0))

            # for img_name in tqdm(test_normal):
            #     img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
            #         (img_size, img_size), resample=Image.BILINEAR)
            #     self.slices.append(img)
            #     self.labels.append(0)
            #     self.img_id.append(img_name.split('.')[0])

            # test abnormal
            # t0 = time.time()
            # self.slices += Parallel(n_jobs=10)(delayed(
            #     lambda file: Image.open(os.path.join(self.root, "train_png_512", file)).convert("L").resize(
            #         (img_size, img_size), resample=Image.BILINEAR))(file) for file in test_abnormal)
            # self.labels += len(test_abnormal) * [0]
            # self.img_id += [img_name.split('.')[0] for img_name in test_abnormal]
            # print("Loaded {} test abnormal images. {:.3f}s".format(len(test_abnormal), time.time() - t0))

            # for img_name in tqdm(test_abnormal):
            #     img = Image.open(os.path.join(self.root, "train_png_512", img_name)).convert('L').resize(
            #         (img_size, img_size), resample=Image.BILINEAR)
            #     self.slices.append(img)
            #     self.labels.append(1)
            #     self.img_id.append(img_name.split('.')[0])

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        return img, label, img_id

    def __len__(self):
        return len(self.slices)
