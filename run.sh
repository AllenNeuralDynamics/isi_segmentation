# Example prediction script 
#!/bin/sh

test_id="661511116_372583_20180207"

python run_predict.py \
    --hdf5_path ./sample_data/${test_id}_processed.hdf5\
    --sign_map_path ./sample_data/${test_id}_sign_map.jpg\
    --label_map_path ./sample_data/${test_id}_label_map.png\
    --model_path ./model/isi_segmentation_model.h5\
    --plot_segmentation True
    
