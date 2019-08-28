import numpy as np
import cv2
import logging
import pickle

logger = logging.getLogger('__main__')

# code adapted from https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26
# TODO Chris suggestions
# TODO different thresholds from OTSU
# TODO remove kernel and recombination steps
# use angles of contours??
# hough transform??

# TODO - countour sorting is currently incorrect and is causing all sorts of fun errors

def process_image(path):
    """

    :param path: path of the image to be processed
    :return: None
    """
    img = cv2.imread(path, 0)
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Invert the image
    img_bin = 255 - img_bin 
    
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80
     
    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=2)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=1)
    
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=2)
    horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=1)
    
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    
    # This function helps to add two image with specific weight
    # parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    # img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1) # makes things worse!
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort by left-to-right, top-to-bottom
    contours = sorted(contours,
                      key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img_final_bin.shape[1])
    
    logger.debug(f'all contours created, total = {len(contours)}')

    cropped_dir_path = r'data/cropped/main_cropped/'
    cropped_row_path = r'data/cropped/row_cropped/'
    cropped_col_path = r'data/cropped/col_cropped/'
    
    ind, r_ind, c_ind = 0, 0, 0
    scale = 0.80
    img = cv2.imread(path, 0)
    ref_img = cv2.imread(path, 0)
    box_contours = []
    
    with open('data/contours.pkl', 'wb') as handle:
        pickle.dump(contours, handle)

    for c in contours:
        # Returns the top left vertex coords and width,height for every contour
        # x measured from left, y measured from top
        box_contours.append(c)
        x, y, w, h = cv2.boundingRect(c)
        centre = (x + w/2, y + h/2)
        
        if (20 < w < 50 and 20 < h < 50):        
            # apply scaling
            crop_w, crop_h = int(w * scale), int(h * scale)
            crop_x, crop_y = int(centre[0] - crop_w//2), int(centre[1] - crop_h//2)
            
            new_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            cv2.imwrite(cropped_dir_path+str(ind) + '.png', new_img)
            cv2.rectangle(img,(crop_x,crop_y),(crop_x+crop_w,crop_y+crop_h),(255,0,0),1)
            cv2.putText(img, str(ind),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            
            # for row inequalities
            if (ind + 1) % 5 != 0:
                logger.debug('row added')
                x_right = int(x + w * 1.1)
                w_right = int(w * 0.7)
                new_img = ref_img[y:y+h, x_right:x_right+w_right]
                
                cv2.imwrite(cropped_row_path+str(r_ind) + '.png', new_img)
                
                r_ind = ind - (ind // 5)
                # cv2.putText(img, str(r_ind),(x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # r_ind += 1
                cv2.rectangle(img,(x_right,y),(x_right+w_right,y+h),(255,0,0),1)
              
            # for column inequalities
            if ind < 20:
                y_below = int(y + h * 1.1)
                h_below = int(h * 0.7)
                new_img = ref_img[y_below:y_below+h_below, x:x+w]
                
                cv2.imwrite(cropped_col_path+str(c_ind) + '.png', new_img)
                c_ind += 1
                cv2.rectangle(img,(x,y_below),(x+w,y_below+h_below),(255,0,0),1)
                
            # if ind == 0: break
                
            ind += 1
            
    logger.debug(f'{ind} final contours kept')
    
    cv2.imwrite('data/img_with_boxes.png', img)

if __name__ == '__main__':
    process_image('data/capture_sample.jpg')