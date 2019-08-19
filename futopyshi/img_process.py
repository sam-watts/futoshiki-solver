import numpy as np
import cv2
import logging

logger = logging.getLogger('__main__')

# code adapted from https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26
# TODO Chris suggestions
# TODO different thresholds from OTSU
# TODO remove kernel and recombination steps
# use angles of contours??
# hough transform??

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
    contours = sorted(contours,
                      key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img_final_bin.shape[1])
    
    logger.debug(f'all contours created, total = {len(contours)}')

    cropped_dir_path = r'data/cropped/'
    idx = 0
    scale = 0.80
    img = cv2.imread(path)
    
    for c in contours:
        # Returns the location and width,height for every contour
        # x measured from left, y measured from top
        x, y, w, h = cv2.boundingRect(c)
    
        # crop bounding rectangles
        centre = (x + w/2, y + h/2)
        w, h = int(w * scale), int(h * scale)
        x, y = int(centre[0] - w//2), int(centre[1] - h//2)
        
        if 20 < w < 50 and 20 < h < 50:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
            # draw as rect for whole image view
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
    logger.info(f'{idx} final contours identified')
    # write whole image with bounding boxes drawn
    cv2.imwrite('data/img_with_boxes.png', img)

if __name__ == '__main__':
    process_image('data/capture_sample.jpg')