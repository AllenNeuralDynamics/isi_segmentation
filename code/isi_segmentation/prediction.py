"""Run inference on a sign map
  
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
import tensorflow as tf
import copy

from datetime import datetime
from isi_segmentation.utils import extract_sign_map_from_hdf5, read_img_forpred, verify_image_shape
from isi_segmentation.postprocess import post_process 
from isi_segmentation.plot import plot_img_label
from isi_segmentation.types import PathLike


def predict(
    hdf5_path: PathLike, 
    sign_map_path: PathLike, 
    label_map_path: PathLike, 
    model_path: PathLike) -> np.ndarray:
    """ Predict the label map for the sign map.
    
    Note that the label map will be saved as '.png' file with different values
    corresponding to different visual cortex areas. 
    The class defination is shown as follows:
    1: VISp,  2: VISam, 3: VISal, 4: VISl, 5: VISrl, 6: VISpl, 7: VISpm, 
    8: VISli, 9: VISpor, 10: VISrll, 11: VISlla, 12: VISmma, 13: VISmmp, 14: VISm,

    Args:
        hdf5_path: path to the hdf5_path which contains the sign map
        sign_map_path: path to save input sign map
        label_map_path: path to save output label map
        model_path: path to the trained isi-segmentation model
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "model_path not a valid file, please download the trained model and update model_path")
    
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError("hdf5_path not a valid file")

    if label_map_path[-4:] != ".png":
        raise NameError("The output label map will be saved as .png file")
    
    #----------------------------------
    # Extract sign map from hdf5 file and save to sign_map_path if it does not exist
    #----------------------------------
    
    print("---" * 20)
    if not os.path.isfile(sign_map_path):
        extract_sign_map_from_hdf5(hdf5_path, sign_map_path)
    
    if not os.path.isfile(sign_map_path):
        raise FileNotFoundError("sign_map_path not a valid file")
    
    print(f"Load the sign map from {sign_map_path}")
    print("---" * 20)

    # Get the input sign map shape
    sign_map = cv2.imread(sign_map_path, cv2.IMREAD_GRAYSCALE) # sign image shape: (540, 640)

    #----------------------------------
    # Read in the sign map for prediction
    #----------------------------------
    
    image = read_img_forpred(sign_map_path)  # resize sign map to shape (512, 512) for prediction 
    verify_image_shape(image.shape, (1, 512, 512))
    
    #----------------------------------
    # Load model and predict on the sign map
    #----------------------------------
    
    model = tf.keras.models.load_model(model_path)

    print("Run prediction ...")
    start_time = datetime.now()
    pred = model.predict(image, verbose=0)[0] 
    end_time = datetime.now()
    print('Prediction duration: {}'.format(end_time - start_time))
        
    pred = np.argmax(pred, axis=-1)

    #----------------------------------
    # Resize and post-process the prediction
    #----------------------------------

    # Resize to original sign map shape
    pred = cv2.resize(pred, (sign_map.shape[1], sign_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = pred.astype(np.int32)
    verify_image_shape(pred.shape, sign_map.shape)
    
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
    verify_image_shape(post_pred.shape, sign_map.shape)
    
    #----------------------------------    
    # Save the label map to label_map_path
    #----------------------------------
    
    print(f"Save the label map to {label_map_path}")
    cv2.imwrite(label_map_path, post_pred)
    
    #----------------------------------
    # Plot results 
    #----------------------------------
    
    savefig_path = label_map_path.replace(".png", "_visualize.png")
    print(f"Plot segmentation, save to {savefig_path}")

    plot_img_label(sign_map_path, 
                  label_map_path, 
                  savefig_path)

    return post_pred
    
