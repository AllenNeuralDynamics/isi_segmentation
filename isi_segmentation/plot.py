""" Helper plot functions"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from isi_segmentation.types import PathLike

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

CLASS_NAME_MAP = {
    0: 'N/A',
    1: 'VISp',
    2: 'VISam',
    3: 'VISal',
    4: 'VISl',
    5: 'VISrl',
    6: 'VISpl',
    7: 'VISpm',
    8: 'VISli',
    9: 'VISpor',
    10: 'VISrll',
    11: 'VISlla',
    12: 'VISmma',
    13: 'VISmmp',
    14: 'VISm',
}

def plot_img_label(
    sign_map_path: PathLike, 
    label_map_path: PathLike, 
    savefig_path: PathLike) -> None:
    """ Visualize the sign map and label map 
    
    Args:
        sign_map_path: path to the sign map
        label_map_path: path to the label map
        savefig_path: path to save plot
    """
    assert os.path.isfile(sign_map_path), "sign_map_path not a valid file"
    assert os.path.isfile(label_map_path), "label_map_path not a valid file"
            
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))    

    #-------------------------------------
    # show sign map
    #-------------------------------------
    
    sign_map = cv2.imread(sign_map_path, cv2.IMREAD_GRAYSCALE) 
    sign_map = sign_map.astype(np.float32)
    
    ax[0].imshow(sign_map, cmap='jet')
    ax[0].set_title("Sign map")
    
    #-------------------------------------
    # show label map
    #-------------------------------------

    label_map = cv2.imread(label_map_path, cv2.IMREAD_GRAYSCALE) 
    label_map = label_map.astype(np.int32)
    
    label_map_3d = np.ndarray(shape=(label_map.shape[0], label_map.shape[1], 3), dtype=int)
    
    for i in range(0, label_map.shape[0]):
        for j in range(0, label_map.shape[1]):
            label_map_3d[i][j] = CLASS_COLOR_MAP[ label_map[i][j] ]    
            
    ax[1].imshow( (label_map_3d/label_map_3d.max() * 255).astype(np.uint8) )
    ax[1].set_title("Label map")
   
    #-------------------------------------
    # overlay cortex name on segmented areas 
    #-------------------------------------
    classes = list(set(label_map.flatten()))

    for cur_class in classes[1:]:
        # get the segmented regions for current class
        mask = np.zeros_like(label_map)
        mask[ label_map == cur_class ] = 1

        # compute the centroid 
        count = (mask == 1).sum()
        y_center, x_center = np.argwhere(mask==1).sum(0)/count
        ax[1].text(
            x_center, 
            y_center, 
            CLASS_NAME_MAP[cur_class], 
            ha="center", 
            va="center", 
            fontsize=8)    

    # plt.show()
    plt.savefig(savefig_path, bbox_inches = 'tight', pad_inches = 0.01)
    plt.close()

   