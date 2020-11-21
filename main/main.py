from capture import get_puzzle
from segmentation_prediction import Predictor
from img_process import process_image
from ocr import scan_images
from solver import solve_puzzle
import logging
import time
import click
import cv2

# debugging
import matplotlib.pyplot as plt


formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(filename)15s:%(lineno)s] --- %(message)s", "%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler('runtime.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(consoleHandler)

def main():
    """Run the end to end solver program.
    
    This will run a loop to continually re-capture images 
    when SPACE is pressed until the solver finds a valid solution
    
    If a GPU is available, this will be used to run real-time inference 
    displayed back on the input image at ~20 FPS
    """

    logger.info('-' * 10 + ' new run ' + '-' * 10)
    
    # load puzzle semantic segmentation model, download from GCS if not done already
    p = Predictor('../models/best_model_3.pth')
    auto_capture = False
    
    while True:
        try:
            image = p.video_overlay(capture=True, pred_size=False, auto_capture=auto_capture) 
            boxes = process_image(image, debug=True)  # process captured image

            start_numbers, inequals = scan_images(boxes) # run ocr

            logger.debug(f'inequals: {inequals}')
            
            print('Solutions:')
            status = solve_puzzle(start_numbers, inequals)

            if status != 0:
                logger.info('-' * 10 + ' run exiting ' + '-' * 10)
                return status
            
        except KeyboardInterrupt:
            raise            
            
        except Exception:
            auto_capture = True
            pass
            
if __name__ == '__main__':
    main()
