import numpy as np
import pytesseract
import glob
import re
import cv2
import os
import logging

logger = logging.getLogger('__main__')

def puzzle_ocr(cropped_path):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    paths = glob.glob(cropped_path)
    puzzle_size = int(np.sqrt(len(paths)))
    
    start_numbers = {}
    for file in paths:
        img = cv2.imread(file)
        # check average coloration combinin all 3 channels
        average = sum(img.mean(axis=0).mean(axis=0))
        
        if average < 740:
            # --psm N
            # Set Tesseract to only run a subset of layout analysis and assume a certain form of image. 
            # The options for N are:
            # 10 = Treat the image as a single character
            val = pytesseract.image_to_string(file, config='--psm 10')
            index = int(re.findall(r'\d+', file)[0]) - 1
            val = int(val)
            start_numbers[index] = val
        
        # os.remove(file)
        
    logger.debug(f'puzzle size = {puzzle_size}')
    logger.debug(f'start numbers = {start_numbers}')
    
    return puzzle_size, start_numbers