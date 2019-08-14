from capture import *
from img_process import *
from ocr import *
from solver import *
import logging

### TODO children of main need to inherit logging config

logging.basicConfig(filename='runtime.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

IMG_PATH = 'data/puzzle_capture.png'
CROPPED_PATH = 'data/cropped/*.png'

def main():
    get_puzzle(IMG_PATH)
    process_image(IMG_PATH)
    puzzle_size, start_numbers = puzzle_ocr(CROPPED_PATH)
    model = create_model(puzzle_size, start_numbers)
    status = solve_model(model)
    
if __name__ == '__main__':
    main()