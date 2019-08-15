from futopyshi.capture import *
from futopyshi.img_process import *
from futopyshi.ocr import *
from futopyshi.solver import *
import logging
import time

fh = logging.FileHandler('runtime.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(filename)15s:%(lineno)s] --- %(message)s", 
                              "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

IMG_PATH = 'data/puzzle_capture.png'
CROPPED_PATH = 'data/cropped/*.png'
SAMPLE_IMG = 'data/sample_img.jpg'


def main():
    """

    :return:
    """

    logger.info('-' * 10 + ' new run ' + '-' * 10)
    # get_puzzle(IMG_PATH) # test processing
    
    # start timer after image is captured
    start = time.time()
    process_image(SAMPLE_IMG)
    
    puzzle_size, start_numbers = puzzle_ocr(CROPPED_PATH)
    
    interval = time.time() - start
    # withold solution if desired! pause timer
    input('Image processed\nPress ENTER to solve!')
    start = time.time()
                  
    status = solve_puzzle(puzzle_size, start_numbers)
    total_time = time.time() - start + interval
    print(f'Processing time: {round(total_time, 4)}')
    
    logger.info('-' * 10 + ' run exiting ' + '-' * 10)
    
    # TODO add cleanup script?
    return status


if __name__ == '__main__':
    main()
