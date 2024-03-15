# Example prediction script 
#!/bin/sh

python predict.py \
    --hdf5_path ./sample_data/661511116_372583_20180207_processed.hdf5\
    --sign_map_path ./sample_data/661511116_372583_20180207_sign_map.jpg\
    --label_map_path ./sample_data/661511116_372583_20180207_label_map.png\
    --model_path ./model/isi_segmentation_model.h5\
    # --plot_segmentation True
