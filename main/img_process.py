import numpy as np
import cv2
import logging
import pickle
from tqdm import tqdm
from typing import Union, Tuple
try:
    from utils import resolve_path
except ImportError:
    from .utils import resolve_path

logger = logging.getLogger('__main__')

def process_image(path: Union[str, np.ndarray], debug: bool = False) ->  Tuple[list, list, list]:
    """Process the captured puzzle image.
    
    This produces crops of all number boxes, and inferred boxes for inequalities

    :param path: path or object of the image to be processed
    :param debug: save cropped images to file for debugging purposes, defaults to False
    :return: start numbers, row inequalities, column inequalities
    """
    if type(path) == np.ndarray:
        img = path
    elif type(path) == str:
        img = cv2.imread(path, 1)
    else:
        raise TypeError('path must be a path to an image or an image array')
        
    img_ref = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    base_img = img.copy()

    kernel_size = int(np.asarray(img.shape).max() * 0.10 // 1)
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1  # make odd
    C = kernel_size // 10
    
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel_size, C)
    
    # Invert the image
    img_bin = 255 - img_bin 
    
    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 70
     
    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=2)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=2)
    
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=2)
    horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=2)
    
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    
    # This function helps to add two image with specific weight
    # parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    # img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1) # makes things worse!
    _, img_final_bin = cv2.threshold(img_final_bin, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite('img_bin.png', img_bin)
        cv2.imwrite('img_final_bin.png', img_final_bin)
    
    contours, _ = cv2.findContours(img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort by left-to-right, top-to-bottom
    contours = sorted(contours,
                      key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img_final_bin.shape[1] // 50)
    
    logger.debug(f'all contours created, total = {len(contours)}')

    cropped_dir_path = resolve_path('data/cropped/main_cropped/')
    cropped_row_path = resolve_path('data/cropped/row_cropped/')
    cropped_col_path = resolve_path('data/cropped/col_cropped/')

    ind, r_ind, c_ind = 0, 0, 0
    scale = 0.80
    box_low = 0.05 * img.shape[0]
    box_high = 0.25 * img.shape[0]
    rect_size = img.shape[0] // 100
    # img = path
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ref_img = img.copy() # cv2.imread(path, 0)
    box_contours, box_images, col_inequalities, row_inequalities = [], [], [], []
    acc = 0
    
    # with open('futopyshi/data/contours.pkl', 'wb') as handle:
    #     pickle.dump(contours, handle)
    
    for c in tqdm(contours, desc='Processing contours...'):
        # Returns the top left vertex coords and width,height for every contour
        # x measured from left, y measured from top
        box_contours.append(c)
        x, y, w, h = cv2.boundingRect(c)
        centre = (x + w/2, y + h/2)
        
        if (box_low < w < box_high and box_low < h < box_high):    
            # apply scaling
            crop_w, crop_h = int(w * scale), int(h * scale)
            crop_x, crop_y = int(centre[0] - crop_w//2), int(centre[1] - crop_h//2)
            
            new_img = base_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            box_images.append(new_img)

            if debug:
                cv2.imwrite(cropped_dir_path+str(ind) + '.png', new_img)
                cv2.rectangle(img_ref,(crop_x,crop_y),(crop_x+crop_w,crop_y+crop_h),(255, 0,0), rect_size)
                cv2.putText(img_ref, str(ind),(x, y), cv2.FONT_HERSHEY_SIMPLEX, rect_size // 2, (255, 0, 0), rect_size)
            
            
            # for row inequalities
            if (ind + 1) % 5 != 0:
                x_right = int(x + w * 1.1)
                w_right = int(w * 0.7)
                new_img = base_img[y:y+h, x_right:x_right+w_right]
                row_inequalities.append(new_img)
                
                if debug:
                    cv2.imwrite(cropped_row_path+str(r_ind) + '.png', new_img)
                    cv2.rectangle(img_ref,(x_right,y),(x_right+w_right,y+h),(0,255,0),rect_size)
                    # cv2.putText(img_ref, str(r_ind),(x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 1)
                
                r_ind = ind - (ind // 5)    
                # r_ind += 1
                
              
            # for column inequalities
            if ind < 20:
                y_below = int(y + h * 1.1)
                h_below = int(h * 0.7)
                new_img = base_img[y_below:y_below+h_below, x:x+w]
                new_img = np.rot90(new_img, axes=(1,0))  # rotate to make them recognisable
                col_inequalities.append(new_img)
                
                if debug:
                    cv2.imwrite(cropped_col_path+str(c_ind) + '.png', new_img)
                    cv2.rectangle(img_ref,(x,y_below),(x+w,y_below+h_below),(0,0,255),rect_size)
                    # cv2.putText(img_ref, str(c_ind),(x, y_below), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 1)

                c_ind += 1
                                
            ind += 1
            
    logger.debug(f'{ind} final contours kept')
    if debug:
        cv2.imwrite('img_with_boxes.png', img_ref)
    
    if ind == 25:
        print('Accepted contours', ind)
    else:
        raise Exception(f'Error - Incorrect number of boxes detected = {ind}. Please retake image')

    return box_images, row_inequalities, col_inequalities
    
    
if __name__ == '__main__':
    process_image('data/IMG_2713.jpg', debug=True)