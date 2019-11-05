"""
Face Aligner class
==================

Align face to well form

This work is inherited and converted from the official OpenVINO toolkit
Original source code can be found here:
    /path/to/inference_engine_vpu_arm/deployment_tools/inference_engine/samples/smart_classroom_demo
"""


import numpy as np
import cv2 as cv

_H, _W = 112., 96.
ref_lm_norm = [
    30.2946 / _W, 51.6963 / _H, 65.5318 / _W, 51.5014 / _H, 48.0252 / _W,
    71.7366 / _H, 33.5493 / _W, 92.3655 / _H, 62.7299 / _W, 92.2041 / _H]

def getTransform(src, dst):
    """get transforming matrix"""
    src = np.array(src, np.float)
    col_mean_src = np.mean(src, axis=0)
    src -= col_mean_src

    dst = np.array(dst, np.float)
    col_mean_dst = np.mean(dst, axis=0)
    dst -= col_mean_dst

    mean, dev_src = cv.meanStdDev(src)
    dev_src = max(dev_src[0], 1.192e-7)
    src /= dev_src[0]

    mean, dev_dst = cv.meanStdDev(dst)
    dev_dst = max(dev_dst[0], 1.192e-7)
    dst /= dev_dst[0]

    w, u, vt = cv.SVDecomp(np.matmul(src.T, dst))
    r = np.matmul(u, vt)

    m = np.zeros((2, 3))
    m[:, 0:2] = r * (dev_dst[0] / dev_src[0])
    m[:, 2] = col_mean_dst.T - np.matmul(m[:, 0:2], col_mean_src.T)

    return m


class FaceAligner:
    """Face Aligner class"""

    @staticmethod
    def align(face_image, landmark):
        """align an input image with given landmark points"""
        H, W = face_image.shape[:2]
        ref_lm = np.zeros((5, 2)) 
        for i in range(5):
            ref_lm[i, 0] = ref_lm_norm[2*i] * W 
            ref_lm[i, 1] = ref_lm_norm[2*i+1] * H

        m = getTransform(ref_lm, landmark)
        aligned_face = cv.warpAffine(face_image, m, (W, H), cv.WARP_INVERSE_MAP)

        return aligned_face 
