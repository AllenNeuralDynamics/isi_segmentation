"""Run inference on a sign map """
import argparse
from isi_segmentation.prediction import predict

if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, default=None, required=True, 
                        help='path to the hdf5 file which contains the testing sign map')
    parser.add_argument('--sign_map_path', type=str, default=None, required=True, 
                        help='path to the sign map')
    parser.add_argument('--label_map_path', type=str, default=None, required=True, 
                        help='path to save the label map')
    parser.add_argument('--model_path', type=str, default=None, required=True, 
                        help='path to the trained isi-segmentation model')
    
    args = parser.parse_args()
    
    # predict the label map for the sign map.
    label_map = predict(**vars(args))
    
