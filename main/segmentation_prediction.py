import torch
import numpy as np
import cv2
from utils import resolve_path, read_image, download_url
from time import sleep
from typing import Union
import os

MODEL_URL = 'https://storage.googleapis.com/futoshiki-solver/segmentation_models/best_model_3.pth'

def to_tensor(x: np.array, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def resize(image: np.array, size: tuple):
    return cv2.resize(image, size)

def preprocess(image: np.array, size: Union[tuple, bool]=None):
    # resize
    if size:
        image = resize(image, size)

    # unit scale
    image = image / 255

    # apply mu, sigma 
    mu = np.array([0.485, 0.456, 0.406])
    sigma = np.array([0.229, 0.224, 0.225])
    image = (image - mu) / sigma

    # transpose to tensor shape
    image = to_tensor(image)

    return image
    

class Predictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(model_path):
            print('Downloading semantic segmentation model')
            download_url(MODEL_URL, model_path)
        
        self.model = torch.load(model_path).to(self.device)
        
    def predict_mask(self, image, pred_size: Union[tuple, bool]):
        """Predict mask for an image"""
        image = preprocess(image, pred_size)
        x_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        pr_mask = self.model.predict(x_tensor).cpu()
        pr_mask = pr_mask.squeeze().numpy().round()
        return pr_mask
    
    def alpha_mask(self, image: np.array, pred_size: Union[tuple, bool], alpha=0.4):
        """Predict mask for an image, and return as green overlay with transparency"""
        mask = self.predict_mask(image, pred_size)
        
        if pred_size:
            mask = resize(mask, image.shape[1::-1])
            
        overlay_mask = mask.copy() * alpha
        overlay_mask = np.expand_dims(overlay_mask, 2)

        # make a green overlay
        green = np.ones(image.shape, dtype=np.float) * (0,1,0)

        # green over original image
        overlay = green * overlay_mask + image * (1.0 - overlay_mask) / 255
        overlay = np.clip(overlay, 0, 1)
        
        return overlay, mask
    
    
    def video_overlay(self, pred_size: Union[tuple, bool]=((320, 480)), delay: int=10, capture: bool=False, auto_capture: bool=False):
        """Display live predictions from webcam feed"""
        cam = cv2.VideoCapture(0)
        
        if self.device == 'cpu':
            print('Overriding delay as CPU inference is slow')
            delay = 100
        
        if capture:
            message = 'Capture mode - press SPACE to capture, ESC to exit'
        else:
            message = 'Testing mode - press ESC to exit'
        
        while True:
            ret, frame = cam.read()

            if not ret:
                break

            overlay, mask = self.alpha_mask(frame, pred_size)
            
            cv2.putText(overlay, message,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if not auto_capture:
                cv2.imshow('Futoshiki Solver Capture', overlay)
            
            k = cv2.waitKey(delay)

            if k % 256 == 27:
                # ESC pressed
                cam.release()
                cv2.destroyAllWindows()
                raise KeyboardInterrupt('Loop cancelled')
                
            elif (k % 256 == 32 and capture) or auto_capture:
                # SPACE pressed or auto_capture set
                # crop image to mask size - format img[y:y+h, x:x+w]
                masked_input = self.mask_input(frame, mask)
                frame = self.largest_countour_crop(frame, masked_input)
#                 if save_img:
#                     if not cv2.imwrite(save_img, image):
#                         raise Exception('Couldn''t save image')
#                         logger.info(f'{save_img} written')
                        
                cam.release()
                cv2.destroyAllWindows()
                return frame
                        
    def mask_input(self, input_image, mask):
        mask = np.dstack([mask] * 3)
        mask = (mask == 1)
        return np.where(mask, input_image, 0)


    def largest_countour_crop(self, image: np.array, mask: np.array):
        # get largest contour - no rotation
        imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = image[y:y+h, x:x+w]
        return image
    
def get_bounded_mask(image: np.array):
    image = read_image(image)
    pred = Predictor(image, resolve_path('../best_model_2.pth'), (320, 320))
    mask = pred.predict_mask()
    # mask = resize(mask, pred.image.shape[::-1])
    mask = pred.rescale_mask(mask)
    mask = pred.mask_original(mask)
    output = pred.largest_countour_crop(mask)
    return output