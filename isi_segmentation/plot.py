""" Helper plot functions"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

   