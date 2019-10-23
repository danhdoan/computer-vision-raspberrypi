import os
import time

import numpy as np
import cv2 as cv

import altusi.config as cfg
from altusi import helper, imgproc
from altusi.logger import Logger
import headvisualizer as hvis
import altusi.visualizer as vis

from altusi.facedetector import FaceDetector
from altusi.headposer import HeadPoser


LOG = Logger(__file__.split('.')[0])


def app(video_path, video_name, show=False, record=False):
    detector = FaceDetector()
    plugin = detector.getPlugin()
    poser = HeadPoser(plugin=plugin)

    cap = cv.VideoCapture(video_path)
    (W, H), FPS = imgproc.cameraCalibrate(cap)
    _, frm = cap.read()
    frm = imgproc.resizeByHeight(frm)
    H, W = frm.shape[:2]
    FRM_MOD = int(1. * FPS / cfg.pFPS + 0.5)
    LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))

    if record:
        time_str = time.strftime(cfg.TIME_FM)
        writer = cv.VideoWriter(video_name+time_str+'.avi',
                cv.VideoWriter_fourcc(*'XVID'), FPS, (W, H))

    cnt_frm = 0
    while cap.isOpened():
        _, frm = cap.read()
        if not _:
            LOG.info('Reached the end of Video source')
            break

        cnt_frm += 1
        if cnt_frm % FRM_MOD: continue
        print(cnt_frm)
        frm = imgproc.resizeByHeight(frm)

        _start_t = time.time()
        scores, bboxes = detector.getFaces(frm)

        if len(bboxes):
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                face_image = frm[y1:y2, x1:x2]
                yaw, pitch, roll = poser.estimatePose(face_image)

                cpoint = [(x1+x2)//2, (y1+y2)//2]
                frm = hvis.draw(frm, cpoint, (yaw, pitch, roll))
        _prx_t = time.time() - _start_t

        frm = vis.plotInfo(frm, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
        frm = cv.cvtColor(np.asarray(frm), cv.COLOR_BGR2RGB)

        if record:
            writer.write(frm)

        if show:
            cv.imshow(video_name, frm)
            key = cv.waitKey(1)
            if key in [27, ord('q')]:
                LOG.info('Interrupted by Users')
                break

    if record:
        writer.release()
    cap.release()
    cv.destroyAllWindows()


def main(args):
    video_link = args.video if args.video else 0
    app(video_link, args.name, args.show, args.record)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Head Pose\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
