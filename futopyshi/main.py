from capture import *
from img_process import *
from ocr import *
from solver import *
from cleanup import *
import logging
import time
import click

fh = logging.FileHandler('runtime.log')

fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(filename)15s:%(lineno)s] --- %(message)s", 
                              "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(consoleHandler)

IMG_PATH = 'data/puzzle_capture.png'
CROPPED_PATH = 'data/cropped/'
SAMPLE_IMG = 'data/sample_img.jpg'

# TODO add click function 

@click.command()
@click.option('--capture', '-c', default=True, help='Recapture image?')
def main(capture):
    """

    :return:
    """

    logger.info('-' * 10 + ' new run ' + '-' * 10)
    
    if capture:
        get_puzzle(IMG_PATH) # test processing
    
    # start timer after image is captured
    start = time.time()
    
    # process_image(SAMPLE_IMG)  # process sample image
    process_image(IMG_PATH)  # process captured image
    
    puzzle_size, start_numbers, inequals = scan_images(CROPPED_PATH)
    
    logger.debug(f'inequals: {inequals}')
    
    interval = time.time() - start
    # withold solution if desired! pause timer
    input('Image processed\nPress ENTER to solve!')
    start = time.time()
                  
    status = solve_puzzle(puzzle_size, start_numbers, inequals)
    total_time = time.time() - start + interval
    print(f'Processing time: {round(total_time, 4)}')
    
    logger.info('-' * 10 + ' run exiting ' + '-' * 10)
    
    # TODO add cleanup script?
    remove_files()
    
    return status


if __name__ == '__main__':
    main()
