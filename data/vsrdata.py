
import os
import glob

from data import common

import numpy as np
import imageio
import random
import torch
import torch.utils.data as data


class VSRData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train

        self.n_image = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_sharp, self.images_blur, self.images_kernel = self._scan()

        self.num_image = len(self.images_sharp)
        print("Number of images to load:", self.num_image)

        if train:
            self.repeat = 1

        if args.process:
            self.data_sharp, self.data_blur, self.data_kernel = self._load(self.num_image)

    def _scan(self):
        """
        Returns a list of image directories
        """
        image_names_blur = sorted(glob.glob(os.path.join(self.dir_image_blur, "*")))
        image_names_gt   = sorted(glob.glob(os.path.join(self.dir_image_gt, "*")))
        image_names_kernel = sorted(glob.glob(os.path.join(self.dir_image_kernel, "*")))

        names_sharp , names_blur, names_kernel = [], [], []
        for image_name in range(len(image_names_blur)):
            image_sharp = [image_names_gt[image_name]]
            image_blur  = [image_names_blur[image_name]]
            image_kernel = [image_names_kernel[image_name]]
            assert len(image_sharp) == len(image_blur)
            names_sharp.append(image_sharp)
            names_blur.append(image_blur)
            names_kernel.append(image_kernel)
            self.n_image.append(len(image_sharp))

        return names_sharp , names_blur, names_kernel

    def _load(self, n_images):
        data_blur = []
        data_sharp = []
        data_kernel = []
        for idx in range(n_images):

            sharps = np.array([imageio.imread(sharp_name) for sharp_name in self.images_sharp[idx]])
            blurs = np.array([imageio.imread(blur_name) for blur_name in self.images_blur[idx]])
            kernels = np.array([imageio.imread(kernel_name) for kernel_name in self.images_kernel[idx]])

            data_blur.append(blurs)
            data_sharp.append(sharps)
            data_kernel.append(kernels)

        return data_sharp, data_blur, data_kernel

    def __getitem__(self, idx):
        if self.args.process:
            blurs, sharps, filenames = self._load_file_from_loaded_data(idx)
        else:
            blurs, sharps, kernels, filenames = self._load_file(idx)

        blurs_list = [blurs[i] for i in range(blurs.shape[0])]
        blurs = np.concatenate(blurs_list , axis=-1)

        sharp_list = [sharps[i] for i in range(sharps.shape[0])]
        sharps = np.concatenate(sharp_list, axis=-1)

        kernel_list = [kernels[i] for i in range(kernels.shape[0])]
        kernels = np.concatenate(kernel_list, axis=-1)

        patches = [self.get_patch(blurs, sharps)]
        blurs = np.array([patch[0] for patch in patches])
        sharps = np.array([patch[1] for patch in patches])

        blurs = torch.cat(torch.split(torch.from_numpy(blurs) , 3 , dim=-1) , dim=0).numpy()
        sharps = torch.cat(torch.split(torch.from_numpy(sharps), 3, dim=-1) , dim=0).numpy()

        blurs = np.array(common.set_channel(*blurs, n_channels=self.args.n_colors))
        sharps = np.array(common.set_channel(*sharps, n_channels=self.args.n_colors))
        kernels = np.expand_dims(kernels, axis=2)
        kernels = np.expand_dims(kernels, axis=0)
        blur_tensors = common.np2Tensor(*blurs,  rgb_range=self.args.rgb_range)
        sharp_tensors = common.np2Tensor(*sharps,  rgb_range=self.args.rgb_range)
        kernel_tensors = common.np2Tensor(*kernels, rgb_range=self.args.rgb_range)

        return torch.stack(blur_tensors), torch.stack(sharp_tensors), torch.stack(kernel_tensors), filenames


    def __len__(self):
        if self.train:
            return len(self.images_sharp) *  self.repeat
        else:
            #
            return sum(self.n_image)

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_image
            #return idx
        else:
            return idx

    def _load_file(self, idx):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array and list of corresponding filenames
        """
        idx = self._get_index(idx)
        if self.train:
            f_sharps = self.images_sharp[idx]
            f_blurs = self.images_blur[idx]
            f_kernels = self.images_kernel[idx]

            start = self._get_index(random.randint(0, self.n_image[idx] - 1))
            filenames = [os.path.splitext(os.path.basename(file))[0] for file in f_sharps[start:start+1]]

            sharps = np.array([imageio.imread(sharp_name) for sharp_name in f_sharps[start:start+1]])
            blurs = np.array([imageio.imread(blur_name) for blur_name in f_blurs[start:start+1]])
            kernels = np.array([imageio.imread(kernel_name) for kernel_name in f_kernels[start:start + 1]])

            w = sharps.shape[2]
            h = sharps.shape[1]
            w_offset = random.randint(0, max(0, w - self.args.patch_size - 1))
            h_offset = random.randint(0, max(0, h - self.args.patch_size - 1))

            sharps = sharps[:, h_offset:h_offset + self.args.patch_size,
                       w_offset:w_offset + self.args.patch_size,:]
            blurs = blurs[:, h_offset:h_offset + self.args.patch_size,
                      w_offset:w_offset + self.args.patch_size,:]
        else:
            f_sharps = self.images_sharp[idx][0:1]
            f_blurs = self.images_blur[idx][0:1]
            f_kernels = self.images_kernel[idx][0:1]
            filenames = [os.path.basename(file.split("/GT")[0]) + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_sharps]
            sharps = np.array([imageio.imread(sharp_name) for sharp_name in f_sharps])
            blurs = np.array([imageio.imread(blur_name) for blur_name in f_blurs])
            kernels = np.array([imageio.imread(kernel_name) for kernel_name in f_kernels])

        return blurs, sharps, kernels, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        if self.train:
            start = self._get_index(random.randint(0, self.n_image[idx] - 1))
            sharps = self.data_sharp[idx][start:start+1]
            blurs = self.data_blur[idx][start:start+1]
            filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.images_sharp[idx]]

        else:
            f_sharps = self.images_sharp[idx][0:1]
            sharps = self.data_sharp[idx][0:1]
            blurs = self.data_blur[idx][0:1]
            filenames = [os.path.basename(file.split("/GT")[0]) + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_sharps]

        return blurs, sharps, filenames

    def get_patch(self, blur, sharp):

        if self.train:
            blur, sharp = common.get_patch(
                blur,
                sharp,
                patch_size=self.args.patch_size,
            )
            if not self.args.no_augment:
                blur, sharp = common.augment(blur, sharp)
        else:
            ih, iw = blur.shape[:2]
            blur = blur[:ih, :iw]
            sharp = sharp[:ih, :iw]
        return  blur, sharp
