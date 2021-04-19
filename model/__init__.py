import os
from importlib import import_module

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.chop = args.chop
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.ckp = ckp
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
    
    def forward(self, *args):
        if self.chop and not self.training:
            return self.forward_chop(*args)
        else:
            return self.model(*args)
    
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False, filename=''):
        target = self.get_model()
        filename = 'model_{}'.format(filename)
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', '{}latest.pt'.format(filename))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', '{}best.pt'.format(filename))
            )

        if self.save_models:
            # if epoch>self.args.epochs-15:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=False, cpu=False):  #
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )

        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_best.pt'),
                    **kwargs
                ),
                strict=False
            )


    def forward_chop(self , x , shave=10):
        b , c , h , w = x.size()
        patch_size=80
        deblur = torch.empty(b , 3 , h , w)
        neigh0 = torch.empty(b , 3 , h , w)
        neigh1 = torch.empty(b , 3 , h , w)

        x_pad = nn.ZeroPad2d(shave)(x)
        for _h in range(shave , h + shave , patch_size):
            for _w in range(shave , w + shave , patch_size):
                input = x_pad[: , : , _h-shave:_h+patch_size+shave , _w-shave:_w+patch_size+shave]
                _deblur , neigh = self.model(input)
                deblur[: , : , _h-shave:_h+patch_size-shave , _w-shave:_w+patch_size-shave] = _deblur[: , : , shave:-shave , shave:-shave]
                neigh0[: , : , _h-shave:_h+patch_size-shave , _w-shave:_w+patch_size-shave] = neigh[0][: , : , shave:-shave , shave:-shave]
                neigh1[: , : , _h-shave:_h+patch_size-shave , _w-shave:_w+patch_size-shave] = neigh[1][: , : , shave:-shave , shave:-shave]

        return deblur , [neigh0 , neigh1]


