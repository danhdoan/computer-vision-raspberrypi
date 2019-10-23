"""
Helper module
============

Support functions for file utilities
"""

import argparse
import os


def getFilename(file_path):
    path, filename = os.path.split(file_path)

    return path, filename


def getFileNameExt(file_path):
    path, filename = getFilename(file_path)

    filename, ext = os.path.splitext(filename)

    return path, filename, ext

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str,
            required=False,
            help='Video Streamming link or Path to video source')
    parser.add_argument('--name', '-n', type=str,
            required=False, default='camera',
            help='Name of video source')
    parser.add_argument('--show', '-s', 
            default=False, action='store_true',
            help='Whether to show the output visualization')
    parser.add_argument('--record', '-r', 
            default=False, action='store_true',
            help='Whether to save the output visualization')
    parser.add_argument('--flip_hor', '-fh',
            required=False, default=False, action='store_true',
            help='horizontally flip video frame')
    parser.add_argument('--flip_ver', '-fv',
            required=False, default=False, action='store_true',
            help='vertically flip video frame')
    args = parser.parse_args()

    return args
