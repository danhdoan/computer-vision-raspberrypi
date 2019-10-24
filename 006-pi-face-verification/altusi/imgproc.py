"""
Imgproc library
===============

Library to support Image processing functions
"""

"""
Revision
--------
    2019, Oct 03:
        - re-add to AltusI version 0.2
"""

import os
import math
import numpy as np
import cv2 as cv


def resizeByHeight(image, height=720):
    """Resize an image given the expected height and keep the original ratio

    Arguments:
    ----------
        image : numpy.array
            input image to resize

    Keyword Arguments:
    ------------------
        height : int (default: 720)
            expected width of output image

    Returns:
    --------
        out_image : numpy.array
            output resized image
    """
    H, W = image.shape[:2]
    width = int(1. * W * height / H + 0.5)
    out_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

    return out_image


def resizeByWidth(image, width=600):
    """Resize an image given the expected width and keep the original ratio

    Arguments:
    ----------
        image : numpy.array
            input image to resize

    Keyword Arguments:
    ------------------
        width : int (default: 600)
            expected width of output image

    Returns:
    --------
        out_image : numpy.array
            output colored image after resized
    """

    H, W = image.shape[:2]
    height = int(H * width / W)
    out_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
    return out_image


def cameraCalibrate(capturer, size=None, by_height=False):
    """Get camera's information like dimension and FPS

    Arguments:
    ----------
        capturer : cv.VideoCapture
            OpenCV-Video capturer object

    Keyword Arguments:
    ------------------
        width : int (default: None)
            width value to resize by width

    Returns:
    --------
        (W, H) : int, int
            dimension of video's frame
        FPS : float
            FPS of the video stream
    """

    fps = capturer.get(cv.CAP_PROP_FPS)

    while True:
        _, frame = capturer.read()
        if _:
            if size:
                if by_height:
                    frame = resizeByHeight(frame, size)
                else:
                    frame = resizeByWidth(frame, size)
            H, W = frame.shape[:2]

            return (W, H), fps

