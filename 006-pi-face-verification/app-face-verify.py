import os
import time

import numpy as np
import cv2 as cv

from altusi import helper, config as cfg 
from altusi.logger import Logger

from altusi.facedetector import FaceDetector
from altusi.facelandmarker import FaceLandmarker
from altusi.facealigner import FaceAligner
from altusi.faceembedder import FaceEmbedder


LOG = Logger('app-face-id')


def getDistance(u, v):
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    norm = np.sqrt(uu * vv) + 1e-6

    return 1 - uv / norm


def app(image1_path, image2_path):
    # initialize Face detector net
    detector = FaceDetector()

    # initialize Face Landmarker net
    landmarker = FaceLandmarker()

    # initialize Face Aligner
    aligner = FaceAligner()

    # intializa Face Embedder
    embedder = FaceEmbedder()
    # ================================================================

    image1 = cv.imread(image1_path)
    image2 = cv.imread(image2_path)
    assert image1 is not None and image2 is not None
    # ================================================================

    _, faces_1 = detector.getFaces(image1)
    _, faces_2 = detector.getFaces(image2)
    assert len(faces_1) and len(faces_2)
    # ================================================================

    x1, y1, x2, y2 = faces_1[0]
    face_image1 = image1[y1:y2, x1:x2]
    lm1 = landmarker.getLandmark(face_image1)
    aligned_face1 = aligner.align(face_image1, lm1)

    x1, y1, x2, y2 = faces_2[0]
    face_image2 = image2[y1:y2, x1:x2]
    lm2 = landmarker.getLandmark(face_image2)
    aligned_face2 = aligner.align(face_image2, lm2)
    # ================================================================

    emb1 = embedder.getEmb(face_image1)
    LOG.info('emb1 shape: {}'.format(emb1.shape))

    emb2 = embedder.getEmb(face_image2)
    LOG.info('emb2 shape: {}'.format(emb2.shape))

    dist = getDistance(emb1, emb2)
    LOG.info('distance: {:.4}'.format(dist))
    # ================================================================


def main(args):
    image1_path = args.image1
    image2_path = args.image2

    app(image1_path, image2_path)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Face Identification\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
