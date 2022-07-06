from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchio.transforms import RandomBiasField, RandomSpike, RandomMotion, RandomGhosting
# import torchio as tio
def unet_mask_weights():
    c = [-22, 0, 23]
    ab = [-73, 0, 73]
    center_offsets = [(z, x, y) for x in ab for y in ab for z in c]
    weight = torch.zeros((155,256,256))
    sample_weight = torch.ones((110,110,110))
    for center in center_offsets:
         weight[155//2 + center[0] - 55:155//2 + center[0] + 55,
                256//2 + center[1] - 55:256//2 + center[1] + 55,
                256//2 + center[2] - 55:256//2 + center[2] + 55] += sample_weight
    return weight, center_offsets

def nvnet_mask_weights():
    c = [-29, 29]
    ab = [-64, 64]
    center_offsets = [(z, x, y) for x in ab for y in ab for z in c]
    weight = torch.zeros((155,256,256))
    sample_weight = torch.ones((96,128,128))
    for center in center_offsets:
         weight[155//2 + center[0] - 48:155//2 + center[0] + 48,
                256//2 + center[1] - 64:256//2 + center[1] + 64,
                256//2 + center[2] - 64:256//2 + center[2] + 64] += sample_weight
    return weight, center_offsets

def deepmedic_mask_weights():
    c = [-22, 0, 23]
    ab = [-73, -44, -15, 14, 43, 73]
    center_offsets = [(z, x, y) for x in ab for y in ab for z in c]
    weight = torch.zeros((75,176,176))
    sample_weight = torch.ones((30,30,30))
    for center in center_offsets:
         weight[75//2 + center[0] - 15:75//2 + center[0] + 15,
                176//2 + center[1] - 15:176//2 + center[1] + 15,
                176//2 + center[2] - 15:176//2 + center[2] + 15] += sample_weight
    return weight, center_offsets

def crop_center(x, size, offset=[0,0,0]):
    if tuple(x.shape[1:]) == size:
        return x
    rand_off = offset
    #### input modes ####
    crop = tuple(((slice(0, x.shape[0], 1)),(slice(0, x.shape[0], 1))))
    #### Volume ####
    crop += tuple(slice(c // 2 - s // 2 + o, c // 2 + s // 2 + s % 2 + o, 1) for c, s, o in zip(x.shape[1:], size,rand_off))
    return x[crop[1:]]

# resample image - sitk.BSpline as interpolator
def resample_image(img, size=[256,256,155]):#
    identity = sitk.Transform(3, sitk.sitkIdentity)
    # compute new spacing
    new_spacing2 = (img.GetSize()[2]/size[2]) * img.GetSpacing()[2]
    new_spacing1 = (img.GetSize()[1]/size[1]) * img.GetSpacing()[1]
    new_spacing0 = (img.GetSize()[0]/size[0]) * img.GetSpacing()[0]
    spacing = (new_spacing0, new_spacing1, new_spacing2)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return sitk.Resample(img, size, identity, sitk.sitkBSpline, origin, spacing, direction)

# resample label - sitk.sitkNearestNeighbor as interpolator
def resample_label(img, size=[256,256,155]):
    print("og spacing", img.GetSpacing())
    identity = sitk.Transform(3, sitk.sitkIdentity)
    # compute new spacing
    new_spacing2 = (img.GetSize()[2]/size[2]) * img.GetSpacing()[2]
    new_spacing1 = (img.GetSize()[1]/size[1]) * img.GetSpacing()[1]
    new_spacing0 = (img.GetSize()[0]/size[0]) * img.GetSpacing()[0]
    spacing = (new_spacing0, new_spacing1, new_spacing2)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return sitk.Resample(img, size, identity, sitk.sitkNearestNeighbor, origin, spacing, direction)

class BRATS(Dataset):
    def __init__(self, path, mode='train', subset=0.05, images=["flair", "t1", "t1ce", "t2"], size=[256,256,155]):
        if path is None:
            RuntimeWarning("Dataset path is not set!")
        self.Path = Path(path)
        self.imgmodes = images
        self.hgg = list(self.Path.glob("HGG/*.npy"))
        self.lgg = list(self.Path.glob("LGG/*.npy"))
        subset_hgg = int(self.hgg.__len__() * subset)
        subset_lgg = int(self.lgg.__len__() * subset)
        subset_hgg_val = int(self.hgg.__len__() * min(subset + 0.1, 1.0))
        subset_lgg_val = int(self.lgg.__len__() * min(subset + 0.1, 1.0))
        if mode == "train":
            self.data = self.hgg[:subset_hgg] + self.lgg[:subset_lgg]
        elif mode == "test":
            self.data = self.hgg[subset_hgg_val:] + self.lgg[subset_lgg_val:]
        elif mode == "val":
            self.data = self.hgg[subset_hgg:subset_hgg_val] + self.lgg[subset_lgg:subset_lgg_val]
        elif mode == "convert":
            for folder in ["HGG","LGG"]:
                self.flairimgs = list(self.Path.glob(folder+'/*/*flair.nii.gz'))
                self.t1imgs = list(self.Path.glob(folder+'/*/*t1.nii.gz'))
                self.t1ceimgs = list(self.Path.glob(folder+'/*/*t1ce.nii.gz'))
                self.t2imgs = list(self.Path.glob(folder+'/*/*t2.nii.gz'))
                self.labels = list(self.Path.glob(folder+'/*/*seg.nii.gz'))
                assert len(self.flairimgs) == len(self.labels)

                self.img = [self.flairimgs,self.t1imgs,self.t1ceimgs,self.t2imgs]
                self.mask = self.labels
                for index in range(0, len(self.labels)):
                    img = []
                    savefile = self.img[0][index].parent.absolute().__str__()
                    for i in range(len(self.img)):
                        tmp = sitk.ReadImage(self.img[i][index].absolute().__str__())
                        tmp = resample_image(tmp, size=size)
                        img.append(sitk.GetArrayFromImage(tmp))

                    label = sitk.ReadImage(self.labels[index].absolute().__str__())
                    label = resample_label(label, size=size)

                    img.append(sitk.GetArrayFromImage(label))
                    img = np.stack(img)
                    np.save(savefile + f"_{size[0]}",img)
            else:
                print("Please use 'train','test','val' or 'convert' as mode")
                return False

    def __len__(self):
        return self.data.__len__() * 155

    def __getitem__(self, item):
        data = np.load(self.data[item // 155].absolute().__str__())

        img = torch.from_numpy(data[:4, item % 155])
        mask = torch.from_numpy(data[-1, item % 155])

        mask[4==mask] = 3
        img = img.float()
        mask = mask.long()

        return img, mask

class BRATS3D(Dataset):
    def __init__(self, path, mode='train', subset=0.05, images=["flair", "t1", "t1ce", "t2"], dim=3, size=[96,128,128], samples=50, index_list=None, augment=False):
        self.Path = Path(path)
        self.dim = dim
        self.size = size
        self.imgmodes = images
        self.hgg = list(self.Path.glob("HGG/*.npy"))
        self.lgg = list(self.Path.glob("LGG/*.npy"))
        subset_hgg = int(self.hgg.__len__() * subset)
        subset_lgg = int(self.lgg.__len__() * subset)
        subset_hgg_val = int(self.hgg.__len__() * min(subset + 0.1, 1.0))
        subset_lgg_val = int(self.lgg.__len__() * min(subset + 0.1, 1.0))
        self.samples = samples
        self.mode = mode
        self.fullAugmentation = None
        self.patchAugmentation = None
        self.index_list=index_list
        if mode == "train":
            self.data = self.hgg[:subset_hgg] + self.lgg[:subset_lgg]
        elif mode == "test":
            self.data = self.hgg[subset_hgg_val:] + self.lgg[subset_lgg_val:]
        elif mode == "val":
            self.data = self.hgg[subset_hgg:subset_hgg_val] + self.lgg[subset_lgg:subset_hgg_val]
        elif mode == "convert":
            for folder in ["HGG","LGG"]:
                self.flairimgs = list(self.Path.glob(folder+'/*/*flair.nii.gz'))
                self.t1imgs = list(self.Path.glob(folder+'/*/*t1.nii.gz'))
                self.t1ceimgs = list(self.Path.glob(folder+'/*/*t1ce.nii.gz'))
                self.t2imgs = list(self.Path.glob(folder+'/*/*t2.nii.gz'))
                self.labels = list(self.Path.glob(folder+'/*/*seg.nii.gz'))
                assert len(self.flairimgs) == len(self.labels)
                self.img = [self.flairimgs,self.t1imgs,self.t1ceimgs,self.t2imgs]
                self.mask = self.labels
                for index in range(0, len(self.labels)):
                    img = []
                    savefile = self.img[0][index].parent.absolute().__str__()
                    for i in range(len(self.img)):
                        tmp = sitk.ReadImage(self.img[i][index].absolute().__str__())
                        tmp = resample_image(tmp, size=[256,256,155])
                        img.append(sitk.GetArrayFromImage(tmp))

                    label = sitk.ReadImage(self.labels[index].absolute().__str__())
                    label = resample_label(label, size=[256,256,155])
                    img.append(sitk.GetArrayFromImage(label))
                    img = np.stack(img)
                    np.save(savefile + "_256",img)
            else:
                print("Please use 'train','test','val' or 'convert' as mode")
                return False

    def __len__(self):
        if self.dim == 2:
            return self.data.__len__() * 155
        elif self.dim == 3 and self.index_list is None:
            return self.data.__len__() * self.samples
        elif self.dim == 3 and self.index_list is not None:
            return self.data.__len__() * len(self.index_list)
        else:
            print("DO NOT support dims other than 2D and 3D")

    def __getitem__(self, item):
        if self.dim == 2:
            data = np.load(self.data[item // 155].absolute().__str__())

            img = torch.from_numpy(data[:4, item % 155])
            mask = torch.from_numpy(data[-1, item % 155])

            mask[4==mask] = 3
            img = img.float()
            mask = mask.long()
        elif self.dim == 3:
            if self.index_list is None:
                data = np.load(self.data[item // self.samples].absolute().__str__())
            else:
                data = np.load(self.data[item // len(self.index_list)].absolute().__str__())
            if self.fullAugmentation is not None:
                data[:4, :], _, data[-1, :] = self.fullAugmentation(data[:4, :],None,data[-1, :])
            if self.mode == "Train":
                offset = [(ds-s)//4 for ds, s in zip(data.shape[1:], self.size)]
            elif self.index_list is not None:
                offset = self.index_list[item % len(self.index_list)]
            else:
                offset = [0,0,0]
            mask = torch.from_numpy(data[-1, :])
            data = crop_center(data, self.size, offset=offset)
            if self.patchAugmentation is not None:
                for augmentation in self.patchAugmentation:
                    data[:4, :], _, data[-1, :] = augmentation(data[:4, :], None, data[-1, :])
            img = torch.from_numpy(data[:4, :])
            mask[4==mask] = 3
            img = img.float()
            mask = mask.long()
        return img, mask, offset


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # print(crop)
    # print(crop[1:])
    # c = [-22, 0, 23]
    # ab = [-73, -44, -15, 14, 43, 73]
    # center_offsets = [(z, x, y) for x in ab for y in ab for z in c]
    # weight = torch.zeros((75,176,176))
    # sample_weight = torch.ones((30,30,30))
    # for center in center_offsets:
    #      weight[75//2 + center[0] - 15:75//2 + center[0] + 15,
    #             176//2 + center[1] - 15:176//2 + center[1] + 15,
    #             176//2 + center[2] - 15:176//2 + center[2] + 15] += sample_weight
    #      print(75//2 + center[0] - 15,75//2 + center[0] + 15,176//2 + center[1] - 15,176//2 + center[1] + 15, 176//2 + center[2] - 15,176//2 + center[2] + 15)
    # print(center_offsets)
    # print(len(center_offsets))
    # print(weight.unique())

    # d = BRATS('D:\\MICCAI_BraTS_2019_Data_Training', mode="convert", subset=0.6)
    # d = BRATS('D:\\MICCAI_BraTS_2019_Data_Training', mode="train", subset=0.6)
    d = BRATS('U:\\MICCAI_BraTS_2019_Data_Training', mode="val", subset=0.6)
    # d = BRATS('C:\\MICCAI_BraTS_2019_Data_Training', mode="test", subset=0.6)
    print(len(d.data))
