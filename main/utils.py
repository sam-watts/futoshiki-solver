import os 
import sys
from typing import List, Union
import numpy as np
import cv2
from PIL import Image, ImageDraw
import functools

def resolve_path(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


def dim_transform(coord, puzzle_size=5):
    """Transform a 1D grid coordinate into a 2D tuple, default 5x5 grid"""
    return coord // puzzle_size, coord % puzzle_size


def print_puzzle(numbers: np.array, inequals: List[tuple]):
    """Print puzzle with numbers and inequalities

    Args:
        numbers (np.array): [description]
        inequals (List): [description]
    """
    # define inequality symbols
    tb = 'ʌ'  # u'\u22C0' # ⋀ v
    bt = 'v'  # u'\u22C1' # ⋁ ʌ
    lr = '<'
    rl = '>'
    
    # create copy to alter
    ineqs = inequals.copy()
    
    print('+---+---+---+---+---+')
    
    for i, row in enumerate(numbers):
        # if i == 1: break
        scope_ineqs = [ineq for ineq in ineqs if ineq[0][0] == i or ineq[1][0] == i]
        
        row = row.tolist()
        # vals = [self.Value(val) for val in row]
        
        # row / col subs are indisces to subsitute for inequalitiess
        col_sep = ['|', ' '] + list(' | '.join(str(x) for x in row)) + [' ', '|']
        col_subs = [4, 8, 12, 16]
        
        row_sep = list('+---+---+---+---+---+')
        row_subs = [2, 6, 10, 14, 18]
        
        # print out inequalities
        for ineq in scope_ineqs:
            lower = ineq[0]
            higher = ineq[1]
            
            if lower[0] == higher[0]:  # if in the same row
                if lower[1] > higher[1]:  # if lower end is to the right
                    col_sep[col_subs[min(lower[1], higher[1])]] = rl
                elif lower[1] < higher[1]:  # if lower end is to the left
                    col_sep[col_subs[min(lower[1], higher[1])]] = lr
                
            elif i != 4 and lower[1] == higher[1]:  # if in the same column, don't need for final row
                if lower[0] > higher[0]:  # if the lower end is on the bottom column
                    row_sep[row_subs[lower[1]]] = bt
                elif lower[0] < higher[0]:  # if the lower end is on top column
                    row_sep[row_subs[lower[1]]] = tb
            
            # remove entry at end of loop to avoid double printing
            ineqs.remove(ineq)
                    
        print(''.join(col_sep))
        print(''.join(row_sep))


def read_image(image: Union[str, np.array]):
    if type(image) == np.ndarray:
        image = image
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif type(image) == str:
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    
    else:
        raise TypeError('path must be a path to an image or an image array')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image_transpose_exif(im: np.array) -> np.array:
    """Transpose smartphone images that mave be incorrectly oriented.

    Apply Image.transpose to ensure 0th row of pixels is at the visual
    top of the image, and 0th column is the visual left-hand side.
    Return the original image if unable to determine the orientation.

    As per CIPA DC-008-2012, the orientation field contains an integer,
    1 through 8. Other values are reserved.

    Parameters
    ----------
    im: PIL.Image
       The image to be rotated.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)