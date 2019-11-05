"""
Emotion Recognition class
===================

Module for Emotion Recognition
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg


class Emotioner:
    """Face Detector class"""

    def __init__(self,
            xml_path=cfg.EMOTION_XML,
            bin_path=cfg.EMOTION_BIN):
        """Initialize Face detector object"""
        self.__net = cv.dnn.readNet(xml_path, bin_path)

        # with NCS support
        self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


    def getEmotion(self, image):
        """Detect faces in an input image with given threshold"""
        H, W = image.shape[:2]
        blob = cv.dnn.blobFromImage(image, size=(64, 64), ddepth=cv.CV_8U)
        self.__net.setInput(blob)
        out = self.__net.forward().reshape(5)

        idx = np.argmax(out)
        
        return idx 
