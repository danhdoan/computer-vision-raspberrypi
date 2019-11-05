"""
Face Embedder class
===================

Face Embedding computation class
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg


class FaceEmbedder:
    """Face embedder class"""

    def __init__(self, 
            xml_path=cfg.FACE_EMB_XML,
            bin_path=cfg.FACE_EMB_BIN):
        """Initialize Face embedder object"""
        self.__net = cv.dnn.readNet(xml_path, bin_path)
        self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


    def getEmb(self, face_image):
        """get embedding from a face image"""
        blob = cv.dnn.blobFromImage(face_image, size=(128, 128), ddepth=cv.CV_8U)
        self.__net.setInput(blob)

        emb = self.__net.forward().reshape(256)

        return emb
