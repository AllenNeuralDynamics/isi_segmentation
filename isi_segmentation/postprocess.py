"""post processing function""" 
import scipy.ndimage as ni
import cv2
import scipy
import numpy as np
import os


def close_open(img: np.ndarray, closeIter: int, openIter: int) -> np.ndarray:
    """ Use open/close operations to eliminate isolated pixels. 

    Args:
        img: input image
        closeIter: iterations for binary closing
        openIter: iterations for binary opening
    Return:
        output image
    """
    # extract uniqule values in img
    img_flat = img.flatten()
    values  = list(set(list(img_flat)))
     
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for val in values:
        cur_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cur_mask[img[:, :] == val] = 1
        cur_mask = scipy.ndimage.binary_fill_holes(cur_mask).astype(int)

        cur_mask = ni.binary_closing(cur_mask, iterations=closeIter)
        cur_mask = ni.binary_opening(cur_mask, iterations=openIter)
    
        out[cur_mask[:, :] == 1] = val

    return out
    
    
def post_process(
    pred:np.ndarray, 
    closeIter: int, 
    openIter: int, 
    pred_dir_prefix: str) -> np.ndarray:
    """ Post-processing operations on the prediction

    Args:
        pred: the model prediction
        closeIter: iterations for binary closing
        openIter: iterations for binary opening
        pred_dir_prefix: path to save the post-processed image 
        
    Return:
        post-processed prediction: numpy array
    """    
    # Perform open/close operations to eliminate isolated pixels.
    pred = close_open(pred, closeIter, openIter)

    # find unique values
    pr_labels = list(set(list(pred.flatten())))
    pr_labels.sort()

    output_label = np.zeros(shape=(pred.shape[0], pred.shape[1]), dtype=np.uint8)

    for cur_class in pr_labels:    
        mask = np.zeros(shape=(pred.shape[0], pred.shape[1]), dtype=np.uint8)
        mask[ pred == cur_class ] = 1
        cur_fig_path = f"{pred_dir_prefix}_class{cur_class}.png"
        cv2.imwrite(cur_fig_path, mask)

        # Read input
        img = cv2.imread(cur_fig_path, cv2.IMREAD_GRAYSCALE)
        
        # Generate intermediate image; use morphological closing to keep parts of the brain together
        inter = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Find largest contour in intermediate image
        cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        
        # Output the largest blob, only keep one blob for each lcass
        tp_out = np.zeros(img.shape, np.uint8)
        cv2.drawContours(tp_out, [cnt], -1, 255, cv2.FILLED)
        tp_out = cv2.bitwise_and(img, tp_out)

        # fill the holes
        tp_out = scipy.ndimage.binary_fill_holes(tp_out).astype(int)
    
        # From Zhuang et al. eLife 2017;6:e18372. DOI: 10.7554/eLife.18372:
        # Patches smaller than 0.00166 mm2 (100 pixels) were discarded.
        if np.sum(tp_out) < 100:
            tp_out = np.zeros(shape=(pred.shape[0], pred.shape[1]))
        
        # update the cortex area for current class
        output_label[ tp_out == 1 ] = cur_class
        output_label = output_label.astype(int)
        
        if os.path.isfile(cur_fig_path):
            os.remove(cur_fig_path)
        
    return output_label




   
    