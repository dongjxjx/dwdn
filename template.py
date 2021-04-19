#!/usr/bin/env python
# -*- coding: utf-8 -*-

def set_template(args):

    if args.template == 'DWDN':

        args.task = 'Deblurring'

        if args.task == "Deblurring":
            args.data_train = 'BLUR_IMAGE'
            args.dir_data = '../TrainingData'
            args.data_test = 'BLUR_IMAGE'
            args.dir_data_test = './TestData'
            args.reset = False
            args.model = "deblur"
            args.test_only = True
            args.pre_train = "./model/model_DWDN.pt"

            args.save = "deblur"
            args.loss = "1*L1"
            args.patch_size = 256
            args.batch_size = 8
            args.grad_clip = 0.5
            if args.test_only:
                args.save = "deblur_test"
            args.save_results = True
            args.save_models = True
            args.no_augment = True


