""" Helper data process functions"""
import h5py
import cv2
import numpy as np
import os
from isi_segmentation.types import PathLike

""" constant variables for prediction"""
# the shape of input of the UNet should be (512, 512)
IMAGE_W = 512
IMAGE_H = 512

def print_arr_inf(array: np.ndarray) -> None:
    """ Print the intensity information given an array """
    print("Intensity info: {:.2f} ± {:.2f}, max={:.2f}, min={:.2f}, median={:.2f}".format(
          np.mean(array), 
          np.std(array), 
          np.max(array), 
          np.min(array), 
          np.median(array))
         )

def Normalized(x: np.ndarray) -> np.ndarray:
    """ Normalize the value of input array to (0, 1) """
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    return normalized


def extract_sign_map_from_hdf5(hdf5_path: PathLike, img_path: PathLike) -> None:
    """ Extract sign map from hdf5 file and save to img_path """
    with h5py.File(hdf5_path, 'r') as hf:
        img = hf['visual_sign'][()]
        
        # the intensity of sign map should be in range of -1.0 and 1.0
        assert np.min(img) >= -1.0
        assert np.max(img) <= 1.0
        
        # after normalization, the intensity of sign map should be in range of 0.0 and 1.0
        img = Normalized(img)
        
        assert np.min(img) >= 0.0
        assert np.max(img) <= 1.0

        img = np.multiply(img, 255).astype(np.uint8)
        cv2.imwrite(img_path, img)
        print(f"Extract sign map from {hdf5_path} and save to {img_path}")

        
def read_img_forpred(image_path: PathLike) -> np.ndarray:
    """ Read and preprocess the sign map. 

    Args:
        image_path: path to input image
    Return:
        numpy array for input image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # image shape: (540, 640)
    image = cv2.resize(image, (IMAGE_W, IMAGE_H)) # image shape: (512, 512)
    image = image/255.0
    
    # the intensity of input sign map should be in range of 0.0 and 1.0 for prediction
    assert np.min(image) >= 0.0
    assert np.max(image) <= 1.0
    
    image = np.expand_dims(image, axis=0) ## [1, H, W]
    image = image.astype(np.float32)
    
    return image

    
def verify_image_shape(input_shape: tuple, expected_shape: tuple) -> None:
    """Verify the image shape """
    if input_shape != expected_shape:
        raise ValueError(
            f"The shape of input image is {input_shape}, not euqal to the expected shape {expected_shape}!")
