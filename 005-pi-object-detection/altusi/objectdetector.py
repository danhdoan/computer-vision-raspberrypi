"""
Object Detector class
===================

Module for Object Detection
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg


class ObjectDetector:
    """Object Detector class"""

    def __init__(self,
            xml_path=cfg.PERSON_DET_XML,
            bin_path=cfg.PERSON_DET_BIN):
        """Initialize Object detector object"""
        self.__net = cv.dnn.readNet(xml_path, bin_path)

        # with NCS support
        self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


    def getObjects(self, image, def_score=0.5):
        """Detect objects in an input image with given threshold"""
        H, W = image.shape[:2]
        blob = cv.dnn.blobFromImage(image, size=(544, 320), ddepth=cv.CV_8U)
        self.__net.setInput(blob)
        out = self.__net.forward()

        bboxes = []
        scores = []
        for det in out.reshape(-1, 7):
            score = float(det[2])
            if score < def_score: continue

            x1 = max(0, int(det[3] * W))
            y1 = max(0, int(det[4] * H))
            x2 = min(W, int(det[5] * W))
            y2 = min(H, int(det[6] * H))

            bboxes.append((x1, y1, x2, y2))
            scores.append(score)
        
        return scores, bboxes
