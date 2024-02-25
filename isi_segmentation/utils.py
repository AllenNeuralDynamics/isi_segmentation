""" Helper data process functions"""
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


""" constant variables for prediction"""
# the shape of input of the UNet should be (512, 512)
IMAGE_W = 512
IMAGE_H = 512

""" constant variables for class and color definition"""
CLASS_COLOR_MAP = {
    0: [256, 256, 256],
    1: [80, 80, 255],
    2: [0, 255, 0],
    3: [255, 165, 0],
    4: [255, 0, 0],
    5: [0, 159, 172],
    6: [255, 255, 0],
    7: [0, 255, 255],
    8: [100, 55, 200],
    9: [66, 204, 255],
    10: [24, 128, 100],
    11: [201, 147, 153],
    12: [200, 109, 172],
    13: [255, 127, 80],
    14: [204, 255, 66]
}

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

def extract_sign_map_from_hdf5(hdf5_path: str, img_path: str) -> None:
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

def read_img_forpred(image_path: str) -> np.ndarray:
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


def plot_img_label(sign_map_path: str, label_map_path: str, savefig_path: str) -> None:
    """ Visualize the sign map and label map 
    
    Args:
        sign_map_path: path to the sign map
        label_map_path: path to the label map
        savefig_path: path to save plot
    """
    assert os.path.isfile(sign_map_path), "sign_map_path not a valid file"
    assert os.path.isfile(label_map_path), "label_map_path not a valid file"
            
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))    

    #----------------------------
    # show sign map
    #----------------------------
    sign_map = cv2.imread(sign_map_path, cv2.IMREAD_GRAYSCALE) 
    sign_map = sign_map.astype(np.float32)
    
    ax[0].imshow(sign_map, cmap='jet')
    ax[0].set_title("Sign map")
    
    #----------------------------
    # show label map
    #----------------------------
    label_map = cv2.imread(label_map_path, cv2.IMREAD_GRAYSCALE) 
    label_map = label_map.astype(np.int32)

    label_map_3d = np.ndarray(shape=(label_map.shape[0], label_map.shape[1], 3), dtype=int)
    
    for i in range(0, label_map.shape[0]):
        for j in range(0, label_map.shape[1]):
            label_map_3d[i][j] = CLASS_COLOR_MAP[ label_map[i][j] ]
            
    ax[1].imshow(label_map_3d)
    ax[1].set_title("Label map")

    plt.savefig(savefig_path, bbox_inches = 'tight', pad_inches = 0.01)
    plt.close()

    
def verify_image_shape(input_shape: tuple, expected_shape: tuple) -> None:
    """Verify the image shape """
    if input_shape != expected_shape:
        raise ValueError(
            f"The shape of input image is {input_shape}, not euqal to the expected shape {expected_shape}!")
