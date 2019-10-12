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


def processFrame(frm, face_detector, face_landmarker):
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

    return frm


def app(video_link, video_name, show, flip_hor, flip_ver):
    video_links = [
        '/home/pi/Videos/crowd-6582.mp4',
        '/home/pi/Videos/india-444.mp4',
        '/home/pi/Videos/paris-2174.mp4',
        '/home/pi/Videos/scotland-21847.mp4',
    ]
    # initialize Face Detection net
    face_detector = FaceDetector()
    LOG.info('Face Detector initialization done')

    # initialize Face Landmark net
    face_landmarker = FaceLandmarker()
    LOG.info('Face Landmarker initialization done')

    # initialize Video Capturer
    cap0 = cv.VideoCapture(video_links[0])
    cap1 = cv.VideoCapture(video_links[1])
    cap2 = cv.VideoCapture(video_links[2])
    cap3 = cv.VideoCapture(video_links[3])
    # (W, H), FPS = imgproc.cameraCalibrate(cap, size=720, by_height=True)
    # LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))

    time_str = time.strftime(cfg.TIME_FM)
    saved_path = 'output.avi'
    writer = cv.VideoWriter(saved_path, cv.VideoWriter_fourcc(*'XVID'), 24, (1280, 720) )

    cnt_frm = 0 
    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
        _0, frm0 = cap0.read()
        _1, frm1 = cap1.read()
        _2, frm2 = cap2.read()
        _3, frm3 = cap3.read()

        if not _0 or not _1 or not _2 or not _3:
            LOG.info('Reached the end of Video source')
            break

        cnt_frm += 1
        frm0 = imgproc.resizeByHeight(frm0, 360)
        frm1 = imgproc.resizeByHeight(frm1, 360)
        frm2 = imgproc.resizeByHeight(frm2, 360)
        frm3 = imgproc.resizeByHeight(frm3, 360)

        frm0 = processFrame(frm0, face_detector, face_landmarker)
        frm1 = processFrame(frm1, face_detector, face_landmarker)
        frm2 = processFrame(frm2, face_detector, face_landmarker)
        frm3 = processFrame(frm3, face_detector, face_landmarker)

        frm = np.zeros((720, 1280, 3))
        frm[:360, :640] = frm0
        frm[:360, 640:] = frm1
        frm[360:, :640] = frm2
        frm[360:, 640:] = frm3
        LOG.info('frm shape: {}'.format(frm.shape))

        cv.imwrite(str(cnt_frm) + '.jpg', frm)
        writer.write(frm)
        LOG.info('Frames processed: {}'.format(cnt_frm))
           
        if show:
            cv.imshow('output', frm)

            key = cv.waitKey(1)
            if key in [27, ord('q')]:
                LOG.info('Interrupted by Users')
                break

    writer.release()
    cap0.release()
    cap1.release()
    cap2.release()
    cap3.release()
    cv.destroyAllWindows()


def main(args):
    video_link = args.video if args.video else 0
    app(video_link, args.name, args.show, args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Face Detection\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
