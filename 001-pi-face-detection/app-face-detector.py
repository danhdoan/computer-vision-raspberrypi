import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger

from altusi.facedetector import FaceDetector

LOG = Logger('app-face-detector')

def app(video_link, video_name, show, flip_hor, flip_ver):
    # initialize Face Detection net
    face_detector = FaceDetector()

    # initialize Video Capturer
    cap = cv.VideoCapture(video_link)
    (W, H), FPS = imgproc.cameraCalibrate(cap)
    LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))

    while cap.isOpened():
        _, frm = cap.read()
        if not _:
            LOG.info('Reached the end of Video source')
            break

        if flip_ver: frm = cv.flip(frm, 0)
        if flip_hor: frm = cv.flip(frm, 1)

        _start_t = time.time()
        scores, bboxes = face_detector.getFaces(frm, def_score=0.5)
        _prx_t = time.time() - _start_t


        if len(bboxes):
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
    LOG.info('Raspberry Pi: Face Detection')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
