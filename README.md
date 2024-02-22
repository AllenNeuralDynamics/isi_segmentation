## Welcome!
This is a repository for segmenting visual cortex areas for sign map. 
The model was trained on about 2000 isi-experiment data using UNet and TensorFlow.

The sign map will be segmented into different regions and 14 cortex areas could be identified.
The output label map will be saved as '.png' file with different value (i.e., 1, 2, 3 ...) 
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
To use the predict-isi-segmentation library, either clone this repository and install the requirements listed in setup.py or install directly with pip.

Install package

```
pip install predict-isi-segmentation
```

The script should take four inputs:

- hdf5_path (str): the hdf5 file which contains the sign map
- sign_map_path (str): the sign map extracted from .hdf5 file for prediction
- label_map_path (str): the output label map for the given sign map
- plot_segmentation (bool): True if plot the resulting label map after inference. False otherwise.

## Download trained model

```
mkdir -p model
gdown https://drive.google.com/uc?id=13ZSmV9CHDon4D7NwoPQTZub1WmSA5bPD -O ./model/model_version1.h5
```

## Run 
To predict the label map for the sample sign map, run:
```
python predict.py \
    --hdf5_path ./sample_data/661511116_372583_20180207_processed.hdf5 \
    --sign_map_path ./sample_data/661511116_372583_20180207_sign_map.jpg \
    --label_map_path ./sample_data/661511116_372583_20180207_label_map.png \
    --plot_segmentation True
```

## Model output directory structure
After running prediction, a directory will be created with the following structure
```console
    /path/to/outputs/
      ├── <experiment_name>.png
      └── <experiment_name>_visualize.png
```      
* `<experiment_name>.png`: prediction from the sign map, the filename is set to `label_map_path`

* `<experiment_name>_label_visualize.png`: visualize the sign map and its resulting label map if `plot_segmentation` is set to `true`.

An example of isi segmentation outputs is `./sample_data/labels/`



<!-- ## Test
Integration test for predict-isi-segmentation using the sample data, run:
```
cd integration_test
python ./integration_test.py
``` -->

## Visualization

If you would like to visualize the output label map, set `plot_segmentation` to True. 
The plot will be saved as `_label_visualize.png` and stored in the same folder as the label map.





