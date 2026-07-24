from torch.utils import data as data
from torchvision.transforms.functional import normalize
import os
import scipy.io as io
import numpy as np
import random
import math
import torch
import torch.nn.functional as F
import cv2

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize_new

@DATASET_REGISTRY.register()
class ContinuousBicubicDownsampleDataset_y(data.Dataset):
    def __init__(self, opt):
        super(ContinuousBicubicDownsampleDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['all_gt_list']

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        all_img_list = []
        for dataset in self.gt_folder:
            dataset_img_list = os.listdir(dataset)
            for img in dataset_img_list:
                img_path = os.path.join(dataset, img)
                all_img_list.append(img_path)

        self.all_img_list = all_img_list

        self.scale_list = opt['scale_list']
        self.lr_size = opt['lr_size']
        self.sample_size = opt['sample_size']

        self.scale_max = self.scale_list[1]
        self.gt_size_max = math.ceil(self.scale_max * self.lr_size)
        self.gt_size = opt['gt_size']
        self.round_mode = opt.get('round_mode', 'ceil')

    def __getitem__(self, index):
        img_path = self.all_img_list[index]

        img_gt = imfromfile(path=img_path, flag='grayscale', float32=True)  #h*w*c, 0-1, ndarray

        guide_img_path = img_path.replace('HR', 'guide').replace('T2', 'T1')
        guide_img = imfromfile(path=guide_img_path, flag='grayscale', float32=True)  # h*w*c, 0-1, ndarray
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
        if guide_img.ndim == 2:
            guide_img = np.expand_dims(guide_img, axis=2)
        h_img_gt, w_img_gt, _ = img_gt.shape

        if len(self.scale_list) == 2:
            scale = float(random.uniform(self.scale_list[0], self.scale_list[1]))
        else:
            scale = random.choice(self.scale_list)


        lr_size = torch.tensor([self.lr_size, self.lr_size])
        # gt_size = torch.tensor([self.gt_size, self.gt_size])

        if self.round_mode == 'ceil':
            gt_size = torch.tensor([math.ceil(scale * lr_size[0]), math.ceil(scale * lr_size[1])])
            # lr_size = torch.tensor([math.ceil(gt_size[0] / scale), math.ceil(gt_size[1] / scale)])
        elif self.round_mode == 'round':
            gt_size = torch.tensor([round(scale * lr_size[0].item()), round(scale * lr_size[1].item())])
            # lr_size = torch.tensor([round(gt_size[0].item() / scale), round(gt_size[1].item() / scale)])

        start_h_crop_gt = random.randint(0, h_img_gt - gt_size[0])
        start_w_crop_gt = random.randint(0, w_img_gt - gt_size[1])

        img_gt = img_gt[start_h_crop_gt:start_h_crop_gt + gt_size[0], start_w_crop_gt:start_w_crop_gt+gt_size[1], :]
        guide_img = guide_img[start_h_crop_gt:start_h_crop_gt + gt_size[0], start_w_crop_gt:start_w_crop_gt + gt_size[1],
                     :]
        scale_modify_h = float(gt_size[0] / lr_size[0])
        scale_modify_w = float(gt_size[1] / lr_size[1])
        img_lq = np.ascontiguousarray(fft_based_downsample(img_3d = img_gt, scale_h = scale_modify_h, scale_w = scale_modify_w))

        scale_modify = torch.tensor([scale_modify_h, scale_modify_w])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_guide = img2tensor([img_gt, img_lq, guide_img], bgr2rgb=False, float32=True)

        if self.sample_size > 0:
            sample_coords = np.random.randint(0,[gt_size[0], gt_size[1]], size=(self.sample_size, 2))
            sample_coords = torch.tensor(sample_coords)
            # Fetching the colour of the pixels in each coordinates
            colour_values = [img_gt[:, coord[0], coord[1]] for coord in sample_coords]
            img_gt = torch.stack(colour_values, dim = 1)

        else:
            sample_coords = None
            # pad_h = self.gt_size_max - gt_size[0]
            # pad_w = self.gt_size_max - gt_size[1]
            # pad_h = 8 - img_gt.shape[1] % 8
            # pad_w = 8 - img_gt.shape[2] % 8
            pad_h = 0
            pad_w = 0
            #pad gt to the maximum size in order to do paraller training
            # img_lq = F.interpolate(img_lq.unsqueeze(0), mode='bicubic', size=img_gt.unsqueeze(0).shape[2:]).squeeze(0)
            # img_lq = F.pad(img_lq, (0, pad_w, 0, pad_h), 'constant', 0)
            # img_gt = F.pad(img_gt, (0, pad_w, 0, pad_h), 'constant', 0)
            # img_guide = F.pad(img_guide, (0, pad_w, 0, pad_h), 'constant', 0)
            img_gt = img_gt.contiguous().view(1, -1).permute(1, 0)
            img_guide = img_guide.contiguous().view(1, -1).permute(1, 0)
            sample_lst = np.random.choice(
                len(img_gt), img_lq.shape[1]*img_lq.shape[2], replace=False)
            # print(img_lq.shape[1]*img_lq.shape[2])
            img_gt = img_gt[sample_lst]
            img_guide = img_guide[sample_lst]
            img_gt = img_gt.view(img_lq.shape[0], img_lq.shape[1], img_lq.shape[2])
            img_guide = img_guide.view(img_lq.shape[0], img_lq.shape[1], img_lq.shape[2])
            gt_size=torch.tensor([img_lq.shape[1], img_lq.shape[2]])

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_guide, self.mean, self.std, inplace=True)
        # from basicsr.utils import imwrite
        # sr_img_save = img_gt.squeeze(0).numpy()
        # sr_img_save = (sr_img_save * 255.0).astype(np.uint8)
        # imwrite(sr_img_save, "gt.png")
        # sr_img_save = img_lq.squeeze(0).numpy()
        # sr_img_save = (sr_img_save * 255.0).astype(np.uint8)
        # imwrite(sr_img_save, "lq.png")
        # sr_img_save = img_guide.squeeze(0).numpy()
        # sr_img_save = (sr_img_save * 255.0).astype(np.uint8)
        # imwrite(sr_img_save, "guide.png")

        if sample_coords is not None:
            return_d = {'gt':img_gt, 'lq':img_lq, 'guide':img_guide, 'sample_coords': sample_coords, 'scale': scale,
                        'gt_size': gt_size, 'scale_modify': scale_modify}
        else:
            return_d = {'gt': img_gt, 'lq': img_lq, 'guide':img_guide, 'scale': scale,
                        'gt_size': gt_size, 'pad_h': pad_h, 'pad_w': pad_w, 'scale_modify': scale_modify}
        return return_d

    def __len__(self):
        return len(self.all_img_list)


def fft_based_downsample(img_3d, scale_h, scale_w):
    img = img_3d[:, :, 0] * 255
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)

    # 计算 k 空间中心
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    new_rows = round(rows // scale_h)
    new_cols = round(cols // scale_w)
    start_row = crow - new_rows // 2
    end_row = start_row + new_rows
    start_col = ccol - new_cols // 2
    end_col = start_col + new_cols

    kspace_cropped = dft_shifted[start_row:end_row, start_col:end_col]

    f_ishift = np.fft.ifftshift(kspace_cropped)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_back_normalized = np.expand_dims(img_back_normalized, axis=2) / 255
    return img_back_normalized