import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger

from altusi.facedetector import FaceDetector
from altusi.facelandmarker import FaceLandmarker

LOG = Logger('app-face-alignment')

h, w = 112., 96.
ref_lm_norm = [
    30.2946 / w, 51.6963 / h, 65.5318 / w, 51.5014 / h, 48.0252 / w,
    71.7366 / h, 33.5493 / w, 92.3655 / h, 62.7299 / w, 92.2041 / h]

def getTransform(src, dst):
    src = np.array(src, np.float)
    col_mean_src = np.mean(src, axis=0)
    src -= col_mean_src

    dst = np.array(dst, np.float)
    col_mean_dst = np.mean(dst, axis=0)
    dst -= col_mean_dst

    mean, dev_src = cv.meanStdDev(src)
    dev_src = max(dev_src[0], 1.192e-7)
    src /= dev_src[0]

    mean, dev_dst = cv.meanStdDev(dst)
    dev_dst = max(dev_dst[0], 1.192e-7)
    dst /= dev_dst[0]

    w, u, vt = cv.SVDecomp(np.matmul(src.T, dst))
    r = np.matmul(u, vt)

    m = np.zeros((2, 3))
    m[:, 0:2] = r * (dev_dst[0] / dev_src[0])
    m[:, 2] = col_mean_dst.T - np.matmul(m[:, 0:2], col_mean_src.T)

    return m

def alignFace(face_image, landmark):
    H, W = face_image.shape[:2]
    ref_lm = np.zeros((5, 2)) 
    for i in range(5):
        ref_lm[i, 0] = ref_lm_norm[2*i] * W 
        ref_lm[i, 1] = ref_lm_norm[2*i+1] * H

    m = getTransform(ref_lm, landmark)
    img = cv.warpAffine(face_image, m, (W, H), cv.WARP_INVERSE_MAP)

    return img


def app(image_path):
    # initialize Face Detection net
    face_detector = FaceDetector()
    LOG.info('Face Detector initialization done')

    # initialize Face Landmark net
    face_landmarker = FaceLandmarker()
    LOG.info('Face Landmarker initialization done')

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        _, image = cap.read()
        if not _:
            LOG.info('Reached the end of Video source')
            break

        image = imgproc.resizeByHeight(image, 720)

        _start_t = time.time()
        scores, bboxes = face_detector.getFaces(image, def_score=0.5)

        landmarks = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            face_img = image[y1:y2, x1:x2]
            landmark = face_landmarker.getLandmark(face_img)

            aligned_img = alignFace(face_img, landmark)
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
        key = cv.waitKey(1)
        if key in [27, ord('q')]:
            LOG.info('Interrupted by Users')
            break
           
    cap.release()
    cv.destroyAllWindows()


def main(args):
    app(args.image)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Face Detection\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
