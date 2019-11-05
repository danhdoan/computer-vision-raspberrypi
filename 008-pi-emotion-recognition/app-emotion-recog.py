import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger

from altusi.facedetector import FaceDetector
from altusi.emotioner import Emotioner

LOG = Logger(__file__.split(',')[0])
EMOTION = ['neutral', 'happy', 'sad', 'surprise', 'anger']


def getPadding(image, bbox):
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox

    w, h = x2-x1, y2-y1
    if h > w: 
        d = (h - w)/2
        x1 = max(0, int(x1 - d))
        x2 = min(W, int(x2 + d))

    return x1, y1, x2, y2


def app(video_link, video_name, show, record, flip_hor, flip_ver):
    # initialize Face Detection net
    face_detector = FaceDetector()

    # initialize Emotioner net
    emotioner = Emotioner()

    # initialize Video Capturer
    cap = cv.VideoCapture(video_link)
    (W, H), FPS = imgproc.cameraCalibrate(cap)
    LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))

    if record:
        time_str = time.strftime(cfg.TIME_FM)
        writer = cv.VideoWriter(video_name+time_str+'.avi',
                cv.VideoWriter_fourcc(*'XVID'), FPS, (W, H))

    while cap.isOpened():
        _, frm = cap.read()
        if not _:
            LOG.info('Reached the end of Video source')
            break

        if flip_ver: frm = cv.flip(frm, 0)
        if flip_hor: frm = cv.flip(frm, 1)

        _start_t = time.time()
        scores, bboxes = face_detector.getFaces(frm, def_score=0.5)
        emos = []
        pbboxes = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = getPadding(frm, bbox)
            face_img = frm[y1:y2, x1:x2]
            emo_idx = emotioner.getEmotion(face_img)
            emos.append(EMOTION[emo_idx])
            pbboxes.append((x1, y1, x2, y2))
        _prx_t = time.time() - _start_t


        if len(bboxes):
            frm = vis.plotBBoxes(frm, pbboxes, emos) 
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
    app(video_link, args.name, args.show, args.record, args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Face Detection')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
