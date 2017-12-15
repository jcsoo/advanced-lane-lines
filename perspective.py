import cv2
import numpy as np

def perspective_transform(vp, tl, bl, out):
    # Symmetric perspective transform based on vanishing point, top left, bottom left

    src = np.array([
        (tl[0], tl[1]), 
        (vp[0] + (vp[0] - tl[0]), tl[1]), 
        (bl[0], bl[1]), 
        (vp[0] + (vp[0] - bl[0]), bl[1])],
    np.float32)
    dst = np.array([(0, 0), (out[0], 0), (0, out[1]), (out[0], out[1])], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

    def warp(self, M, img):
        return cv2.warpPerspective(img, M, self.img_shape, flags=cv2.INTER_LINEAR)

    def unwarp(self, M, img):
        return cv2.warpPerspective(img, M, self.img_shape, flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_AP)
