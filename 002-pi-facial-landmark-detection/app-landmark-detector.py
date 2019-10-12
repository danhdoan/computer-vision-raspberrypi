import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger

from altusi.facedetector import FaceDetector
from altusi.facelandmarker import FaceLandmarker

LOG = Logger('app-landmark-detector')


def app(video_link, video_name, show, flip_hor, flip_ver):
    # initialize Face Detection net
    face_detector = FaceDetector()
    LOG.info('Face Detector initialization done')

    # initialize Face Landmark net
    face_landmarker = FaceLandmarker()
    LOG.info('Face Landmarker initialization done')

    # initialize Video Capturer
    cap = cv.VideoCapture(video_link)
    (W, H), FPS = imgproc.cameraCalibrate(cap, size=720, by_height=True)
    LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))

    while cap.isOpened():
        _, frm = cap.read()
        if not _:
            LOG.info('Reached the end of Video source')
            break

        if flip_ver: frm = cv.flip(frm, 0)
        if flip_hor: frm = cv.flip(frm, 1)

        frm = imgproc.resizeByHeight(frm, 720)

        _start_t = time.time()
        scores, bboxes = face_detector.getFaces(frm, def_score=0.5)

        landmarks = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            face_img = frm[y1:y2, x1:x2]
            landmark = face_landmarker.getLandmark(face_img)
            landmark[:, :] += np.array([x1, y1])
            landmarks.append(landmark)
        _prx_t = time.time() - _start_t

        if len(bboxes):
            for i, landmark in enumerate(landmarks):
                for j, point in enumerate(landmark):
                    cv.circle(frm, tuple(point), 3, (0, 255, 0), -1)

            frm = vis.plotBBoxes(frm, bboxes, len(bboxes) * ['face'], scores)

        frm = vis.plotInfo(frm, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
        frm = cv.cvtColor(np.asarray(frm), cv.COLOR_BGR2RGB)
       
        if show:
            cv.imshow(video_name, frm)
            key = cv.waitKey(1)
            if key in [27, ord('q')]:
                LOG.info('Interrupted by Users')
                break

    cap.release()
    cv.destroyAllWindows()


def main(args):
    video_link = args.video if args.video else 0
    app(video_link, args.name, args.show, args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Face Detection\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
