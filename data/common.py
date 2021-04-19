import random
import numpy as np
import skimage.color as sc
import torch
"""
Repository for common functions required for manipulating data
"""
def get_patch(*args, patch_size=256):
    ih, iw, c = args[0].shape

    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)

    ret = []

    for arg in args:
        if arg.shape == (ih , iw , c):
            ret.append(arg[iy:iy + patch_size, ix:ix + patch_size, :])
        else:
            ih_sharp, iw_sharp, _ = arg.shape
            ix_sharp = random.randrange(0, iw_sharp - patch_size + 1)
            iy_sharp = random.randrange(0, ih_sharp - patch_size + 1)
            ret.append(arg[iy_sharp:iy_sharp + patch_size, ix_sharp:ix_sharp + patch_size, :])
    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = sc.rgb2ycbcr(img)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):

        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = np.rot90(img)
        
        return img

    return [_augment(a) for a in args]

