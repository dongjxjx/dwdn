# Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring

This repository is the PyTorch implementation of the paper:

**Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring**

Jiangxin Dong, Stefan Roth, and Bernt Schiele

To appear at NeurIPS 2020 (**Oral Presentation**)

[[Paper]](https://proceedings.neurips.cc/paper/2020/file/0b8aff0438617c055eb55f0ba5d226fa-Paper.pdf)/[[Supplemental]](https://proceedings.neurips.cc/paper/2020/file/0b8aff0438617c055eb55f0ba5d226fa-Supplemental.pdf)

## Introduction

We present a simple and effective approach for non-blind image deblurring, combining classical techniques and deep learning. In contrast to existing methods that deblur the image directly in the standard image space, we propose to perform an explicit deconvolution process in a feature space by integrating a classical Wiener deconvolution framework with learned deep features. A multi-scale feature refinement module then predicts the deblurred image from the deconvolved deep features, progressively recovering detail and small-scale structures. The proposed model is trained in an end-to-end manner and evaluated on scenarios with both simulated and real-world image blur. Our extensive experimental results show that the proposed deep Wiener deconvolution network facilitates deblurred results with visibly fewer artifacts. Moreover, our approach quantitatively outperforms state-of-the-art non-blind image deblurring methods by a wide margin.

![Pipeline](https://gitlab.mpi-klsb.mpg.de/jdong/dwdn/raw/master/images/pipeline5.png)

>Deep Wiener deconvolution network. While previous work mostly relies on a deconvolution in the image space, our network first extracts useful feature information from the blurry input image and then conducts an explicit Wiener deconvolution in the (deep) feature space through Eqs. (3) and (8). A multi-scale encoder-decoder network progressively restores clear images, with fewer artifacts and finer detail. The whole network is trained in an end-to-end manner.

## Requirements

Compatible with Python 3

Main requirements: PyTorch 1.1.0 or 1.5.0 or 0.4.1 are tested

To install requirements:

```setup
pip install torch==1.1.0 torchvision==0.3.0
pip install -r requirements.txt
```

## Evaluation

To evaluate the deep Wiener deconvolution network on test examples, run:

```eval
python main.py
```

## Pre-trained Model

Please download the model from https://gitlab.mpi-klsb.mpg.de/jdong/dwdn/-/blob/master/model/model_DWDN.pt and put it in the folder "./model/".

## Bibtex

Please cite our paper if it is helpful to your work:

```
@article{dong2020deep,
  title={Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring},
  author={Dong, Jiangxin and Roth, Stefan and Schiele, Bernt},
  journal={Advances in Neural Information Processing Systems},
  pages = {1048--1059},
  volume={33},
  year={2020}
}
```
