import os
import time

import re
import numpy as np
import cv2 as cv

from tflite_runtime.interpreter import Interpreter

from altusi import Logger, imgproc, helper
from altusi import config as cfg, visualizer as vis

LOG = Logger(__file__.split('.')[0])


def load_labels(path):
    labels = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            _ = line.index(' ')
            idx, cls = int(line[:_]), line[_:].strip()
            labels[idx] = cls

    return labels


def set_input_tensor(interpreter, image):
    tensor_idx = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_idx)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, idx):
    output_details = interpreter.get_output_details()[idx]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, frame_size, thresh=0.5):
    H, W = frame_size 
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    bboxes = get_output_tensor(interpreter, 0)
    class_ids = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    idxes = np.where(scores >= thresh)

    for i, bbox in enumerate(bboxes):
        bboxes[i][0], bboxes[i][1] = bboxes[i][1], bboxes[i][0]
        bboxes[i][2], bboxes[i][3] = bboxes[i][3], bboxes[i][2]

    return scores[idxes], class_ids[idxes], bboxes[idxes] * np.array((W, H, W, H))


def app(image_path):

    labels = load_labels(cfg.LABEL_PATH)
    
    interpreter = Interpreter(cfg.DETECT_MODEL_PATH)
    interpreter.allocate_tensors()
    _, input_H, input_W, _ = interpreter.get_input_details()[0]['shape']

    for i in range(5):
        image = cv.imread(image_path)
        assert image is not None, 'Invalid image path'

        _start_t = time.time()
        inp_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        inp_image = cv.resize(inp_image, (input_W, input_H))
        scores, class_ids, bboxes = detect_objects(interpreter, 
                inp_image, image.shape[:2], thresh=0.6)
        _prx_t = time.time() - _start_t
        LOG.info('FPS: {:.3f}'.format(1/_prx_t))

        image = vis.plotBBoxes(image, bboxes.astype('int'), 
                classes=[labels[idx] for idx in class_ids],
                scores=scores) 
        image = vis.plotInfo(image, 'Raspberry Pi - FPS: {:.2f}'.format(1/_prx_t))
        image = cv.cvtColor(np.asarray(image), cv.COLOR_BGR2RGB)

        cv.imwrite('output.jpg', image)


def main(args):
    app(args.image)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Object Detection with TFLite\n')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
