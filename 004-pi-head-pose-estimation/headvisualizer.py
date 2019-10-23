import numpy as np
import cv2 as cv

scale = 50 

def draw(image, cpoint, headpose):
    H, W = image.shape[:2]

    yaw, pitch, roll = headpose

    yaw *= np.pi / 180.
    pitch *= np.pi / 180.
    roll *= np.pi / 180.

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]])

    Ry = np.array([
        [np.cos(yaw), 0, -np.sin(yaw)],
        [0, 1, 0],
        [np.sin(yaw), 0, np.cos(yaw)]])

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])

    r = np.matmul(Rx, np.matmul(Ry, Rz))

    cameraMatrix = np.array([
        [950., 0, W / 2],
        [0, 950., H / 2],
        [0, 0, 1]])

    xAxis = np.array([1 * scale, 0, 0]).reshape((3, 1))
    yAxis = np.array([0, -1 * scale, 0]).reshape((3, 1))
    zAxis = np.array([0, 0, -1 * scale]).reshape((3, 1))
    zAxis1 = np.array([0, 0, 1 * scale]).reshape((3, 1))

    o = np.array([0, 0, 950.]).reshape((3, 1))

    xAxis = np.matmul(r, xAxis) + o
    yAxis = np.matmul(r, yAxis) + o
    zAxis = np.matmul(r, zAxis) + o
    zAxis1 = np.matmul(r, zAxis1) + o

    p1, p2 = [0, 0], [0, 0]
    p2[0] = xAxis[0, 0] / xAxis[2, 0] * cameraMatrix[0, 0] + cpoint[0]
    p2[1] = xAxis[1, 0] / xAxis[2, 0] * cameraMatrix[1, 1] + cpoint[1]
    p2[0], p2[1] = int(p2[0]), int(p2[1])
    cv.line(image, tuple(cpoint), tuple(p2), (0, 0, 255), 2)

    p2[0] = yAxis[0, 0] / yAxis[2, 0] * cameraMatrix[0, 0] + cpoint[0]
    p2[1] = yAxis[1, 0] / yAxis[2, 0] * cameraMatrix[1, 1] + cpoint[1]
    p2[0], p2[1] = int(p2[0]), int(p2[1])
    cv.line(image, tuple(cpoint), tuple(p2), (0, 255, 0), 2)

    p1[0] = zAxis1[0, 0] / zAxis1[2, 0] * cameraMatrix[0, 0] + cpoint[0]
    p1[1] = zAxis1[1, 0] / zAxis1[2, 0] * cameraMatrix[1, 1] + cpoint[1]
    p1[0], p1[1] = int(p1[0]), int(p1[1])

    p2[0] = zAxis[0, 0] / zAxis[2, 0] * cameraMatrix[0, 0] + cpoint[0]
    p2[1] = zAxis[1, 0] / zAxis[2, 0] * cameraMatrix[1, 1] + cpoint[1]
    p2[0], p2[1] = int(p2[0]), int(p2[1])

    cv.line(image, tuple(p1), tuple(p2), (255, 0, 0), 2)
    cv.circle(image, tuple(p2), 3, (255, 0, 0), 2)

    return image
