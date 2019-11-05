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

def app(image_path):
    # initialize Face Detection net
    face_detector = FaceDetector()

    # initialize Emotioner net
    emotioner = Emotioner()

    frm = cv.imread(image_path)
    _start_t = time.time()
    scores, bboxes = face_detector.getFaces(frm, def_score=0.5)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        face_img = frm[y1:y2, x1:x2]
        emo_idx = emotioner.getEmotion(face_img)
        LOG.info('emotion: {}'.format(emo_idx))
    _prx_t = time.time() - _start_t


#    if len(bboxes):
#        frm = vis.plotBBoxes(frm, bboxes, len(bboxes) * ['face'], scores)
#    frm = vis.plotInfo(frm, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
#    frm = cv.cvtColor(np.asarray(frm), cv.COLOR_BGR2RGB)
       
    cv.destroyAllWindows()


def main(args):
    app(args.image)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Emotion Recognition\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
