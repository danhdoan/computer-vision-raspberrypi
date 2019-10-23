import numpy as np
import cv2 as cv

import altusi.config as cfg
from altusi.inference import Network

class HeadPoser:
    def __init__(self,
            xml_path=cfg.HEAD_POSE_XML,
            device='MYRIAD',
            inp_size=1, out_size=3,
            num_requests=2, plugin=None):
        self.__net = Network()
        self.plugin, (self.B, self.C, self.H, self.W) = self.__net.load_model(
                xml_path, device, inp_size, out_size, num_requests, plugin=plugin)


    def estimatePose(self, face_image):
        image = cv.resize(face_image, (self.W, self.H))
        image = image.transpose((2, 0, 1))
        image = image.reshape((self.B, self.C, self.H, self.W))

        self.__net.exec_net(0, image)
        self.__net.wait(0)

        angle_y = self.__net.get_output(0, 'angle_y_fc')[0, 0]
        angle_p = self.__net.get_output(0, 'angle_p_fc')[0, 0]
        angle_r = self.__net.get_output(0, 'angle_r_fc')[0, 0]

        return angle_y, angle_p, angle_r

    def getPlugin(self):
        return self.plugin
