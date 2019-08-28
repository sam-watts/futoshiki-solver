import numpy as np
import pytesseract
import glob
import re
import cv2
import logging
from general import dim_transform

logger = logging.getLogger('__main__')

# TODO ocr for inequality detection
# within grid, matching to < and > is good - build up a list of characters
# for ^ and upside down version
# eg. ^ in ('A', 'n')
# upside down ^ in ('v', 'V')


def ocr_run(cropped_path, rot=False):
    """
    
    """
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    paths = glob.glob(cropped_path + '*.png')
    num_files = int(np.sqrt(len(paths)))
    
    characters = {}
    for file in paths:
        img = cv2.imread(file, 0)
        # _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # rotate if inequals are veritcal
        if rot: img = np.rot90(img, axes=(1,0))
            
        # check average coloration combine all 3 channels
        # average = sum(img.mean(axis=0).mean(axis=0))
        
        # for one channel
        # average = img.mean()
        # max_img = img.max()
        var = np.var(img)
        
        logger.debug(f'var = {var}')
        
        if var > 80:
            # --psm N
            # Set Tesseract to only run a subset of layout analysis and assume a certain form of image. 
            # The options for N are:
            # 10 = Treat the image as a single character
            
            _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            val = pytesseract.image_to_string(img, config='--psm 10')
            index = int(re.findall(r'\d+', file)[0]) # - 1
            characters[index] = val
        
        # os.remove(file)
        
    logger.debug(f'puzzle size = {num_files}')
    logger.debug(f'start numbers = {characters}')
    
    return num_files, characters

def parse_inequals(inequals, rot=False):
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

    
def scan_images(cropped_path):
    puzzle_size, start_numbers = ocr_run(cropped_path + 'main_cropped/')
    _, row_inequals = ocr_run(cropped_path + 'row_cropped/')
    _, col_inequals = ocr_run(cropped_path + 'col_cropped/', rot=True)
    
    start_numbers = {k: int(v) for k, v in start_numbers.items()}
    
    row_inequals = parse_inequals(row_inequals)
    col_inequals = parse_inequals(col_inequals, rot=True)
        
    inequals = convert_inequals(row_inequals, col_inequals)
    
    return puzzle_size, start_numbers, inequals


def convert_inequals(row_inequals, col_inequals):
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
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img = cv2.imread(path)
    val = pytesseract.image_to_string(img, config='--psm 10')
    val = val[0] 
    
    print('character is: ', val)

if __name__ == '__main__':
    scan_image('data/ineqs/1.png')
