from capture import get_puzzle
from segmentation_prediction import get_bounded_mask
from img_process import process_image
from ocr import scan_images
from solver import solve_puzzle
# from cleanup import remove_files
import logging
import time
import click
import cv2

# debugging
import matplotlib.pyplot as plt


formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(filename)15s:%(lineno)s] --- %(message)s", 
                              "%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler('runtime.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(consoleHandler)

IMG_PATH = 'data/puzzle_capture.png'
CROPPED_PATH = 'data/cropped/'
SAMPLE_IMG = 'data/IMG_2713.jpg'

IMG_PATH = SAMPLE_IMG

# TODO refactor code so that models etc. are all loaded at the start 

@click.command()
@click.option('--capture', '-c', is_flag=True, default=False, help='Recapture image?')
def main(capture):
    """Run the end to end solver program"""

    logger.info('-' * 10 + ' new run ' + '-' * 10)
    
    if capture:
        image = get_puzzle(IMG_PATH) # test processing
    else:
        image = cv2.imread(IMG_PATH)
    
    # start timer after image is captured
    start = time.time()
    
    # image = get_bounded_mask(image) # for semantic segmentation
    # plt.imsave('seg_check.png', image)

    boxes = process_image(image, debug=True)  # process captured image
    
    start_numbers, inequals = scan_images(boxes)
    
    logger.debug(f'inequals: {inequals}')
    
    interval = time.time() - start
    
    # withold solution if desired! pause timer
    input('Image processed\nPress ENTER to solve!')
    start = time.time()
                  
    status = solve_puzzle(start_numbers, inequals)
    total_time = time.time() - start + interval
    print(f'Processing time: {round(total_time, 4)}')
    
    logger.info('-' * 10 + ' run exiting ' + '-' * 10)
    

    
    return status


if __name__ == '__main__':
    main()
