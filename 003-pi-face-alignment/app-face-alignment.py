import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger

from altusi.facedetector import FaceDetector
from altusi.facelandmarker import FaceLandmarker
from altusi.facealigner import FaceAligner

LOG = Logger('app-face-alignment')


def app(image_path):
    # initialize Face Detection net
    face_detector = FaceDetector()
    LOG.info('Face Detector initialization done')

    # initialize Face Landmark net
    face_landmarker = FaceLandmarker()
    LOG.info('Face Landmarker initialization done')

    # initialize Face Alignment class
    face_aligner = FaceAligner()
    LOG.info('Face Aligner initialization done')

    # initialize Video Capturer
    image = cv.imread(image_path)
    assert image is not None

    image = imgproc.resizeByHeight(image, 720)

    _start_t = time.time()
    scores, bboxes = face_detector.getFaces(image, def_score=0.5)

    landmarks = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        face_img = image[y1:y2, x1:x2]
        landmark = face_landmarker.getLandmark(face_img)

        aligned_img = face_aligner.align(face_img, landmark)
        cv.imshow('aligned-faces' + str(i), aligned_img)

        landmark[:, :] += np.array([x1, y1])
        landmarks.append(landmark)
    _prx_t = time.time() - _start_t

    if len(bboxes):
        for i, landmark in enumerate(landmarks):
            for j, point in enumerate(landmark):
                cv.circle(image, tuple(point), 3, (0, 255, 0), -1)

        image = vis.plotBBoxes(image, bboxes, len(bboxes) * ['face'], scores)

    image = vis.plotInfo(image, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
    image = cv.cvtColor(np.asarray(image), cv.COLOR_BGR2RGB)

    cv.imshow(image_path, image)
    key = cv.waitKey(0)
       
    cv.destroyAllWindows()


def main(args):
    app(args.image)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Face Alignment\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
