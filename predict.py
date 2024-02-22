"""Run inference on a sign map
The sign map will be segmented into different regions.
The output label map will be saved as '.png' file with different value 
corresponds to different visual cortex areas. 
The class definition is as follows:
1: VISp 
2: VISam 
3: VISal 
4: VISl
5: VISrl 
6: VISpl
7: VISpm
8: VISli
9: VISpor
10: VISrll 
11: VISlla
12: VISmma
13: VISmmp 
14: VISm
  
The flow of prediction is as follows:
- input: the sign map
- step 0: extract sign map from .hdf5 file if it does not exist
- step 1: read and preprocess the sign map
- step 2: load the trained model and run prediction on the given sign map
- step 3: post-process the prediction: remove isolated pixels, only keep one patch per class, 
          discard patches smaller than 100 pixels.
- step 4: save the final label map

"""

import cv2
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import argparse
import copy
from isi_segmentation.utils import (
    extract_sign_map_from_hdf5,
    read_img_forpred,
    plot_img_label,
)

from isi_segmentation.postprocess import post_process 


def predict(hdf5_path, sign_map_path, label_map_path, model_path, plot_segmentation=False):
    """ Predcit the label map for the sign map.
    
    Note that the label map will be saved as '.png' file with different value 
    corresponds to different visual cortex areas in the label map. 
    The class defination is shown as follows:
    1: VISp;  2: VISam; 3: VISal; 4: VISl; 5: VISrl; 6: VISpl; 7: VISpm; 
    8: VISli; 9: VISpor; 10: VISrll; 11: VISlla; 12: VISmma; 13: VISmmp; 14: VISm; 

    Args:
        hdf5_path (str): path to the hdf5_path which contains the sign map
        sign_map_path (str): path to save input sign map
        label_map_path (str): path to save output label map
        model_path (str): path to the trained isi-segmentation model
        plot_segmentation (bool): True if plot the resulting label map after inference. False otherwise.
        
    Return:
        numpy array for input image
    """
    assert os.path.isfile(hdf5_path), "hdf5_path not a valid file"
    assert label_map_path[-4:] == ".png", "The output label map will be saved as .png file"   
    assert os.path.isfile(model_path), "model_path not a valid file, please download the trained model and update model_path"
     
    #----------------------------------
    # Extract sign map from hdf5 file and save to sign_map_path
    #----------------------------------
    
    print("---" * 20)
    print(f"Extract sign map from {hdf5_path}")
    extract_sign_map_from_hdf5(hdf5_path, sign_map_path)
        
    assert os.path.isfile(sign_map_path), "sign_map_path not a valid file"   
    
    print(f"Load the sign map from {sign_map_path}")
    print("---" * 20)

    # Get the input sign map shape
    sign_map = cv2.imread(sign_map_path, cv2.IMREAD_GRAYSCALE) # sign image shape: (height, width), (540, 640)

    #----------------------------------
    # Read in the sign map for prediction
    #----------------------------------
    
    image = read_img_forpred(sign_map_path)  # resize sign map to shape (512, 512) for prediction 
    assert image.shape == (1, 512, 512), f"The shape of input image is {image.shape}."
    
    #----------------------------------
    # Load model and predict on the sign map
    #----------------------------------

    model = tf.keras.models.load_model(model_path)

    print("Run prediction ...")
    pred = model.predict(image, verbose=0)[0] 
    pred = np.argmax(pred, axis=-1)

    #----------------------------------
    # Resize and post-process the prediction
    #----------------------------------

    # Resize to original sign map shape
    pred = cv2.resize(pred.astype(float), (sign_map.shape[1], sign_map.shape[0])) 
    pred = pred.astype(np.int32)
    
    assert pred.shape == sign_map.shape

    # Post-process the output label map
    print("Run post-processing ...")
    closeIter = 5 
    openIter  = 5
    # path to save intermediate images
    pred_dir_prefix = label_map_path.replace(".png", "")
    post_pred = post_process(pred, 
                             closeIter, 
                             openIter, 
                             pred_dir_prefix)
    
    assert post_pred.shape == sign_map.shape

    #----------------------------------    
    # Save the label map to label_map_path
    #----------------------------------
    
    print(f"Save the label map to {label_map_path}")
    cv2.imwrite(label_map_path, post_pred)
    
    #----------------------------------
    # Plot results if plot_segmentation is set to true
    #----------------------------------
    
    if plot_segmentation == True:
        savefig_path = label_map_path.replace(".png", "_visualize.png")
        print(f"Plot segmentation, save to {savefig_path}")
        
        plot_img_label(sign_map_path, 
                      label_map_path, 
                      savefig_path)

    
if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, default=None, required=True, 
                        help='path to the hdf5 file which contains the testing sign map')
    parser.add_argument('--sign_map_path', type=str, default=None, required=True, 
                        help='path to save the sign map')
    parser.add_argument('--label_map_path', type=str, default=None, required=True, 
                        help='path to save the label map')
    parser.add_argument('--model_path', type=str, default=None, required=True, 
                        help='path to the trained isi-segmentation model')
    parser.add_argument('--plot_segmentation', type=bool, default=False, 
                        help='plot segmentation after inference?')
    
    args = parser.parse_args()
    
    predict(**vars(args))