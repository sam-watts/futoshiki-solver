import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

logger = logging.getLogger('__main__')

def capture_image(save_img: str=None, flip: bool=False) -> Union[np.ndarray, bool]:
    """Capture webcam image of puzzle using cv2 display.

    :param save_img: path to write image to, defaults to None
    :return: captured image
    """
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        height, width = frame.shape[:2]
        grid_length = 300 
        
        tl = (width // 2 - grid_length // 2, height // 2 - grid_length // 2)  # top left
        br = (width // 2 + grid_length // 2, height // 2 + grid_length // 2)  # bottom right
        
        output = frame.copy()
        
        if flip:
            output = cv2.flip(output, -1)
            
        cv2.rectangle(output, tl, br, (255, 0, 0), 2)
        cv2.putText(output, 'Align puzzle with grid, then press SPACE BAR to capture image',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Futoshiki Solver Capture', output)
        
        if not ret:
            break
            
        k = cv2.waitKey(1)
    
        if k % 256 == 27:
            # ESC pressed
            logger.info('Escape hit, closing window')
            cv2.destroyAllWindows()
            return False
            
        elif k % 256 == 32:
            # SPACE pressed
            # crop image to frame size - format img[y:y+h, x:x+w]
            frame = frame[tl[1]:tl[1]+grid_length, tl[0]:tl[0]+grid_length]
            image = cv2.flip(frame, 0)
            if save_img:
                if not cv2.imwrite(save_img, image):
                    raise Exception('Couldn''t save image')
                    logger.info(f'{save_img} written')
                
            cam.release()
            return image    


def display_preview(image: np.ndarray) -> bool:
    """Display preview of a captured image and return the choice
    
    If ESC is pressed - take another image
    If SPACE pressed - accept image
    
    :param image: captured image to be previewed
    :return: confirmation of captured image show in preview window
    """    
    text = 'Press SPACE BAR to accept image\nPress ESC to take another'
    y0, dy = 30, 20
    
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        img_ref = image.copy()
        cv2.putText(img_ref, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Futoshiki Solver Capture', img_ref)
    
    k = cv2.waitKey()
    
    if k % 256 == 27:
        # ESC pressed, take another
        logger.info('Retake image')
        return False
        
    elif k % 256 == 32:
        # SPACE pressed, accept image
        logger.info('Image accepted')
        return True


def get_puzzle(save_img: str = None) -> Union[np.ndarray, bool]:
    """Capture a puzzle image with confirmation of the captured image to be used.

    :param save_img: path to write image to, defaults to None
    :return: either the image captured, or False if the capture process has been cancelled
    """
    cv2.namedWindow('Futoshiki Solver Capture')
    
    while True:
        image = capture_image(save_img)

        # break if no image taken
        print(type(image))
        if type(image) != np.ndarray:
            break

        image_confirmed = display_preview(image)
        
        if image_confirmed:
            cv2.destroyAllWindows()
            break

    return image

if __name__ == '__main__':
    get_puzzle('data/capture_sample.jpg')