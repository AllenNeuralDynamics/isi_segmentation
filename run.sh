# Example prediction script 
#!/bin/sh
# mkdir if not exists
# mkdir -p model
# gdown https://drive.google.com/uc?id=13ZSmV9CHDon4D7NwoPQTZub1WmSA5bPD -O ./model/model_version1.h5

python predict.py \
    --hdf5_path ./sample_data/661511116_372583_20180207_processed.hdf5 \
    --sign_map_path ./sample_data/661511116_372583_20180207_sign_map.jpg \
    --label_map_path ./sample_data/661511116_372583_20180207_label_map.png \
    --model_path ./model/model_version1.h5
    --plot_segmentation True