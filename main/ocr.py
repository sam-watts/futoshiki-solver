import numpy as np
import pytesseract
import glob
import re
import cv2
import logging
# from general import dim_transform
from utils import print_puzzle, dim_transform
from typing import Tuple

# debugging
import matplotlib.pyplot as plt

logger = logging.getLogger('__main__')

# TODO use pytorch model instead of tesseract

def ocr_run(images: list, rot: bool=False, char_whitelist: str=None) -> dict:
    """Run the OCR process on a list of images
    
    :param images: images to detect from
    :param rot: whether to rotate the image by 90 degrees, defaults to False
    :param char_whitelist: the pool of characters to predict from for the OCR engine
    :return: values of processes characters, with key as original location in list
    """
    num_images = int(np.sqrt(len(images)))
    all_var = [np.var(x) for x in images]
    mean_var = np.sum(all_var) / len(images)

    std_var = np.sqrt(np.var(all_var))
    check_var = mean_var + 0.5 * std_var

    characters = {}
    for i, image in enumerate(images):
        # rotate if inequals are vertical
        if rot: 
            image = np.rot90(image, axes=(1,0))

        # get image variance to decide whether or not to detect from it
        var = all_var[i]
        
        logger.debug(f'image index = {i} | var = {var} | check_var = {check_var}')
        
        if var > check_var:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # psm 10 = treat the image as a single character
            config = f'--psm 10 -c tessedit_char_whitelist={char_whitelist} --tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata\"'
            val = pytesseract.image_to_string(image, config=config)
            characters[i] = val
        
    logger.debug(f'start numbers = {characters}')
    
    return characters

def parse_inequals(inequals:dict, rot: bool = False) -> dict:
    """Parse inequalities of different types, performing rotation if needed"""
    if rot:
        l = 'bottom_lower'
        r = 'top_lower'
        
    else:
        l = 'left_lower'
        r = 'right_lower'
    
    for k, v in inequals.items():
        if v[0].lower() in ('<'):
            inequals[k] = l
        else:
            inequals[k] = r

    return inequals

    
def scan_images(boxes: Tuple[list, list, list]) -> Tuple[dict, dict]:
    """Extract numbers, row inequalities and column inequalities
    
    :param boxes: images of numbers and two types of inequalities
    :return: starting numbers and inequality values and locations
    """
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    start_numbers = ocr_run(boxes[0], char_whitelist='12345')
    row_inequals = ocr_run(boxes[1], char_whitelist='<>')
    col_inequals = ocr_run(boxes[2], char_whitelist='<>')
    
    # start_numbers = {2: '2', 14: '4', 20: '3'} # TESTING

    try:
        start_numbers = {k: int(v) for k, v in start_numbers.items()}
    except:
        raise ValueError(f'Invalid start numbers\n{start_numbers}')
    
    row_inequals = parse_inequals(row_inequals)
    col_inequals = parse_inequals(col_inequals, rot=True)
        
    inequals = convert_inequals(row_inequals, col_inequals)
    
    # puzzle size fixed to 5 X 5
    numbers = np.zeros((5, 5))
    for k, v in start_numbers.items():
        row = k // 5
        col = k - (row * 5)
        numbers[row, col] = v

    numbers = numbers.astype(int)
    print_puzzle(numbers, inequals)
    
    return start_numbers, inequals


def convert_inequals(row_inequals: dict, col_inequals: dict) -> dict:
    """Convert inequalities into solver format"""
    inequals = []
    
    for k, v in col_inequals.items():
        above = dim_transform(k) 
        below = dim_transform(k + 5)
        if v == 'top_lower':
             inequals.append((above, below))
        elif v == 'bottom_lower':
             inequals.append((below, above))
             
    
    for k, v in row_inequals.items():
        adjust = k // 4
        left = dim_transform(k + adjust)
        right = dim_transform(k + adjust + 1)
        
        if v == 'left_lower':
            inequals.append((left, right))
        elif v == 'right_lower':
            inequals.append((right, left))
            
    return inequals
    
    

def scan_image(path):
    """Testing function to scan a single image from file"""
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    img = cv2.imread(path)
    # img = 255 - img 
    val = pytesseract.image_to_string(img, lang='eng', config='--psm 10 -c tessedit_char_whitelist=12345<> --tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata\"')
    val = val[0] 
    
    print('character is: ', val)

if __name__ == '__main__':
    scan_image('data/cropped/main_cropped/2.png')
