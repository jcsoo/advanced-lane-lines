import cv2
import numpy as np
import sys, os, pickle

from calibrate import ImageLoader

PIPELINE_CFG = {
    'calibration_path': 'calibration.p',
    'mask' : (0.08, 0.40),
}

class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loader = ImageLoader(cfg['calibration_path'])

    def load_image(self, path):
        return self.loader.load_bgr(path)

    def mask_image(self, img, mask):        
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        if mask is None:
            return img

        imshape = img.shape

        mask_w, mask_h = mask

        vertices = np.array([[
            (0,imshape[0]),
            (imshape[1] * (0.5 - mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1] * (0.5 + mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1], imshape[0])
        ]], dtype=np.int32)


        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image        

    def run(self, path):
        img = self.load_image(path)
        img_orig = img.copy()

        img = self.mask_image(img, self.cfg.get('mask', None))

        return img

    def view(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()   


def main(args):
    pipeline = Pipeline(PIPELINE_CFG)

    for arg in args:
        pipeline.view(pipeline.run(arg))


if __name__ == '__main__':
    main(sys.argv[1:])