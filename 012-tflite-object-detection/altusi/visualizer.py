"""
Visualization module
====================

Process Image visualization, PIL package is in use
"""

"""
Revision
--------
    2019, Sep 25: first version
        - add plotBBoxes (with class ids)
"""


import random as rnd

import numpy as np
import cv2 as cv
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageColor

import altusi.config as cfg

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

COLOR_MAP = {
    'face':'Crimson', 'bicycle':'BlueViolet', 'bus':'Gold', 'car':'DodgerBlue', 
    'motorbike':'OrangeRed', 'person':'Chartreuse'
}

def getRandomColor():
    """Generate random color
    
    Returns:
    --------
        color : tuple(int, int, int)
            generated random color
    """

    color = tuple([int(255*rnd.random()) for _ in range(3)])
    return color


def plotBBoxes(image, bboxes, classes=None, scores=None, color='Chartreuse', linewidth=2, use_rgb=False):
    """Plot bounding boxes for given input objects

    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        bboxes : list((x1, y1, x2, y2))
            input bounding boxes of objects to draw

    Keyword Arguments:
    ------------------
        classes : list(str) (default: None)
            list of classes for objects 
        color : str (default: `Chartreuse`) 
            color to plot
        linewidth : int (default: 2)
            how thick the shape is
        use_rgb : bool (default: False)
            whether using RGB image or not (apply for NDArray image)

    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    if len(bboxes) == 0:
        return image
        exit()

    if isinstance(image, np.ndarray):
        if not use_rgb:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    W, H = image.size
    draw = ImageDraw.Draw(image)

    if classes is not None:
        font = ImageFont.truetype(font=cfg.FONT, 
                    size=np.floor(3e-2*min(H, W) + 0.5).astype('int32') )
        
        for i, ((x1, y1, x2, y2), cls) in enumerate(zip(bboxes, classes)):
            # draw bounding box
            if cls in COLOR_MAP:
                draw.rectangle([x1, y1, x2, y2], 
                               outline=ImageColor.getrgb(COLOR_MAP[cls]),
                               width=linewidth)
            else:
                draw.rectangle([x1, y1, x2, y2], 
                               outline=ImageColor.getrgb('Chartreuse'),
                               width=linewidth)

            # draw label
            text = cls
            if scores is not None:
                text = '{} {:.2f}'.format(cls, scores[i])

            label_size = draw.textsize(text, font) 
            text_coor = np.array([x1+linewidth, max(0, y1 - label_size[1] - 1)])
            rec_coor = np.array([x1, text_coor[1]])

            if cls in COLOR_MAP:
                draw.rectangle([tuple(rec_coor), 
                               tuple(rec_coor + label_size + np.array([linewidth*2, 0])) ], 
                               fill=ImageColor.getrgb(COLOR_MAP[cls]))
            else:
                draw.rectangle([tuple(rec_coor), 
                               tuple(rec_coor + label_size + np.array([linewidth*2, 0])) ], 
                               fill=ImageColor.getrgb('Chartreuse'))
            draw.text(text_coor, text, fill=ImageColor.getrgb('black'), font=font)
    else:
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            draw.rectangle([x1, y1, x2, y2], 
                           outline=ImageColor.getrgb(color),
                           width=linewidth)

    del draw
    return image


def plotInfo(image, info, color='Chartreuse', use_rgb=False):
    if isinstance(image, np.ndarray):
        if not use_rgb:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    W, H = image.size
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font=cfg.FONT, 
                size=np.floor(5e-2*min(H, W) + 0.5).astype('int32'))

    label_size = draw.textsize(info, font)
    text_coor = np.array([5, 0])
    rec_coor = np.array([0, 0])
    draw.rectangle([tuple(rec_coor), 
                   tuple(rec_coor + label_size + np.array([10, 0])) ], 
                   fill=ImageColor.getrgb(color))
    draw.text(text_coor, info, fill=ImageColor.getrgb('black'), font=font)

    del draw
    return image
