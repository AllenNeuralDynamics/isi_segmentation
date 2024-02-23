## Welcome!
This is a repository for segmenting visual cortex areas for sign map. 
The model was trained on about 2000 isi-experiment data using UNet and TensorFlow.

The sign map will be segmented into different regions and 14 cortex areas could be identified.
The output label map will be saved as '.png' file with different values (i.e., 1, 2, 3 ...) 
corresponding to different visual cortex areas (i.e., VISp, VISam, VISal ...). 
The class definition is as follows:  
| Class | acronym | name | 
| :---------- | :----------- | :------------ |
| 1 | VISp | Primary visual area |
| 2 | VISam | Anteromedial visual area |
| 3 | VISal | Anterolateral visual area |
| 4 | VISl | Lateral visual area |
| 5 | VISrl | Rostrolateral visual area |
| 6 | VISpl | Posterolateral visual area |
| 7 | VISpm | posteromedial visual area |
| 8 | VISli | Laterointermediate area |
| 9 | VISpor | Postrhinal area |
| 10 | VISrll | Rostrolateral lateral visual area |
| 11 | VISlla | Laterolateral anterior visual area |
| 12 | VISmma | Mediomedial anterior visual area |
| 13 | VISmmp | Mediomedial posterior visual area |
| 14 | VISm | Medial visual area |


## Installation
To use the predict-isi-segmentation library, clone this repository and install the requirements.

1. First, ensure git is installed:
```
git --version
```
If `git` is not recognized, install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).


2. Move into the directory where you want to place the repository folder, and then download it from GitHub:

```
cd <SOME_FOLDER>
git clone https://github.com/Di-Wang-AIND/predict_isi_segmentation.git
cd predict_isi_segmentation
pip install -r requirements.txt
```

<!-- pip install predict-isi-segmentation -->


The script should take four inputs:

- hdf5_path (str): the hdf5 file which contains the sign map
- sign_map_path (str): the sign map extracted from .hdf5 file for prediction
- label_map_path (str): the output label map for the given sign map
- model_path (str): path to trained model (to download it, follow [here](#Download-trained-model))

## Download trained model

```
mkdir -p model
gdown 'https://drive.google.com/uc?id=13ZSmV9CHDon4D7NwoPQTZub1WmSA5bPD' -O ./model/isi_segmentation_model.h5
```

## Run 
To predict the label map for the sample sign map with the download model, run:
```
python predict.py \
    --hdf5_path ./sample_data/661511116_372583_20180207_processed.hdf5\
    --sign_map_path ./sample_data/661511116_372583_20180207_sign_map.jpg\
    --label_map_path ./sample_data/661511116_372583_20180207_label_map.png\
    --model_path ./model/isi_segmentation_model.h5
```

Or you could directly run 
```
sh run.sh
```

Please make sure you have already downloaded the trained model (follow [here](#Download-trained-model)) and update `model_path`. 


## Model output directory structure
After running prediction, a directory will be created with the following structure
```console
    /path/to/outputs/
      ├── <experiment_name>.png
      └── <experiment_name>_visualize.png
```      
* `<experiment_name>.png`: prediction from the sign map, the filename is set to `label_map_path`

* `<experiment_name>_visualize.png`: visualize the sign map and its resulting label map if `plot_segmentation` is set to `true`.

An example of isi segmentation outputs is `./sample_data/labels/`


## Visualization

If you would like to visualize the output label map, set `plot_segmentation` to True. 
The plot will be saved as `<experiment_name>_visualize.png` and stored in the same folder as the label map.





