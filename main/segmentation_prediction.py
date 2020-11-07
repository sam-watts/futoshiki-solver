import torch
import numpy as np
import cv2
from utils import resolve_path, read_image

def to_tensor(x: np.array, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_model(path: str):
    return torch.load(path)

def resize(image: np.array, size: tuple):
    return cv2.resize(image, size)

def preprocess(image: np.array, size: tuple):
    # resize
    image_vis = resize(image, size)

    # unit scale
    image_pred = image_vis / 255

    # apply mu, sigma 
    mu = np.array([0.485, 0.456, 0.406])
    sigma = np.array([0.229, 0.224, 0.225])
    image_pred = (image_pred - mu) / sigma

    # transpose to tensor shape
    image_pred = to_tensor(image_pred)

    return image_pred
    

class Predictor:
    def __init__(self, image, model_path, size):
        self.image = image
        self.model = get_model(model_path)
        self.image_pred = preprocess(self.image, size)

    def predict_mask(self):
        x_tensor = torch.from_numpy(self.image_pred).unsqueeze(0)
        pr_mask = self.model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().numpy().round()
        return pr_mask
    
    def rescale_mask(self, mask: np.array):
        return resize(mask, self.image.shape[1::-1])  
    
    def mask_original(self, mask: np.array):
        mask = np.dstack([mask] * 3)
        mask = (mask == 1)
        return np.where(mask, self.image, 0)

    def largest_countour_crop(self, mask: np.array):
        # get largest contour - no rotation
        imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = self.image[y:y+h, x:x+w]
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