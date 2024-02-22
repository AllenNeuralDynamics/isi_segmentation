# Example prediction script 
#!/bin/sh

python predict.py \
    --hdf5_path ./sample_data/processed_hdf5/661511116_372583_20180207_processed.hdf5 \
    --sign_map_path ./sample_data/sign_maps/661511116_372583_20180207_processed.jpg \
    --label_map_path ./sample_data/labels/661511116.png \
    --plot_segmentation Flase