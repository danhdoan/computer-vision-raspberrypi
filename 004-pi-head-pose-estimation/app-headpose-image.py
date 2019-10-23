import os
import time

import numpy as np
import cv2 as cv

import altusi.config as cfg
from altusi import helper
from altusi.logger import Logger
import headvisualizer as hvis
import altusi.visualizer as vis

from altusi.facedetector import FaceDetector
from altusi.headposer import HeadPoser


LOG = Logger(__file__.split('.')[0])


def app(image_path):
    detector = FaceDetector()
    plugin = detector.getPlugin()
    poser = HeadPoser(plugin=plugin)
    image = cv.imread(image_path)

    scores, bboxes = detector.getFaces(image)

    if len(bboxes):
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            face_image = image[y1:y2, x1:x2]
            yaw, pitch, roll = poser.estimatePose(face_image)

            LOG.info('yaw: {:.4f} pitch: {:.4f} roll: {:.4f}'.format(yaw, pitch, roll))

            cpoint = [(x1+x2)//2, (y1+y2)//2]
            image = hvis.draw(image, cpoint, (yaw, pitch, roll))


    cv.imshow(image_path, image)
    cv.waitKey(0)


def main(args):
    app(args.image)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Head Pose\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
