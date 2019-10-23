"""
Facial Landmarker class
=======================

Module for Facial Landmark Detection
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg


class FaceLandmarker:
    """Face Landmarker class"""

    def __init__(self,
            xml_path=cfg.FACE_LM_XML,
            bin_path=cfg.FACE_LM_BIN):
        """Initialize Face Landmarker object"""
        self.__net = cv.dnn.readNet(xml_path, bin_path)
        self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


    def getLandmark(self, image):
        """Locate Facial landmark points in a face image"""
        H, W = image.shape[:2]

        blob = cv.dnn.blobFromImage(image, size=(48, 48), ddepth=cv.CV_8U)
        self.__net.setInput(blob)

        reg = self.__net.forward().reshape(-1, 10)
        points = np.zeros((5, 2), np.int)
        for i  in range(0, 10, 2):
            points[i//2] = (reg[0][i:i+2] * np.array([W, H])).astype(np.int)

        return points

