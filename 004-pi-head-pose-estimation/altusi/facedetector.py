"""
Face Detector class
===================

Module for Face Detection
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg

from altusi.inference import Network


class FaceDetector:
    """Face Detector class"""

    def __init__(self,
            xml_path=cfg.FACE_DET_XML, 
            device='MYRIAD', 
            inp_size=1, out_size=1, 
            num_requests=2, plugin=None):
        """Initialize Face detector object"""
        self.__net = Network()
        self.plugin, (self.B, self.C, self.H, self.W) = self.__net.load_model(
                xml_path, device, inp_size, out_size, num_requests, plugin=plugin)


    def getFaces(self, image, def_score=0.5):
        """Detect faces in an input image with given threshold"""
        H, W = image.shape[:2]

        frm = cv.resize(image, (self.W, self.H))
        frm = frm.transpose((2, 0, 1))
        frm = frm.reshape((self.B, self.C, self.H, self.W))

        self.__net.exec_net(0, frm)
        self.__net.wait(0)

        res = self.__net.get_output(0)

        scores, bboxes = [], []
        for obj in res[0][0]:
            if obj[2] > def_score:
                score = obj[2]
                xmin = max(0, int(obj[3] * W))
                ymin = max(0, int(obj[4] * H))
                xmax = min(W, int(obj[5] * W))
                ymax = min(H, int(obj[6] * H))

                scores.append(score)
                bboxes.append([xmin, ymin, xmax, ymax])

        return scores, bboxes


    def getPlugin(self):
        return self.plugin
