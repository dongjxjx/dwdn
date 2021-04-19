import os
from data import vsrdata

# Data loader for blur images
class BLUR_IMAGE(vsrdata.VSRData):

    def __init__(self, args, name='BLUR_IMAGE', train=True):
        super(BLUR_IMAGE, self).__init__(args, name=name, train=train)

    def _scan(self):
        names_sharp, names_blur, names_kernel = super(BLUR_IMAGE, self)._scan()

        return names_sharp, names_blur, names_kernel
        
    def _set_filesystem(self, dir_data):

        if self.args.template == "DWDN" :
            print("loading image...")
            self.apath = os.path.join(dir_data)
            self.dir_image_blur = os.path.join(self.apath, "InputBlurredImage")
            self.dir_image_gt = os.path.join(self.apath, "InputTargetImage")
            self.dir_image_kernel = os.path.join(self.apath, "psfMotionKernel")
            print("Training data path:" , self.dir_image_blur)

