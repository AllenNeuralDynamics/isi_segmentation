import h5py
import numpy as np
import math
import os
import json
import sys
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import json
import imageio
from dataclasses import dataclass

def eccentricity(az, alt, az_center, alt_center):
    """Compute the eccentricity map given azimuth, altitude, and their centers."""
    daz = az - az_center
    dalt = alt - alt_center
    ecc = np.arctan(
        np.sqrt(
            np.square(np.tan(dalt)) +
            np.square(np.tan(daz)) / np.square(np.cos(dalt))
        )
    )
    return ecc

def retinotopy_metric(mask, map):
    """Calculate min, max, range, and bias for a retinotopy map within a mask."""
    ind = np.where(mask > 0)
    vals = map[ind]
    maxv = np.degrees(np.max(vals))
    minv = np.degrees(np.min(vals))
    return (minv, maxv, maxv - minv, abs(minv + maxv))

def window_level(input, cmin, cmax):
    """Apply window leveling to an input array between cmin and cmax."""
    data = np.copy(input)
    crange = cmax - cmin
    data[data < cmin] = cmin
    data[data > cmax] = cmax
    data -= cmin
    data /= (cmax - cmin)
    return data

def outline_mask(mask):
    """Generate an outline mask from a binary mask using Sobel edge detection."""
    edge_horizont = ndimage.sobel(mask, 0)
    edge_vertical = ndimage.sobel(mask, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    outline = np.zeros(mask.shape)
    outline[magnitude > 0] = 1
    outline = np.uint8(outline)
    return outline

@dataclass
class ISIData:
    date: any
    retinotopy_altitude: np.ndarray
    retinotopy_azimuth: np.ndarray
    visual_sign: np.ndarray
    vasculature_image: np.ndarray
    defocus_image: np.ndarray
    visual_sign_pixel_size: float
    vasculature_pixel_size: float
    retinotopy_altitude_shape: tuple
    retinotopy_azimuth_shape: tuple
    visual_sign_shape: tuple
    vasculature_image_shape: tuple
    defocus_image_shape: tuple
    label_map_image: np.ndarray = None

    @classmethod
    def from_files(cls, hdf5_file_path, label_map_path=None):
        with h5py.File(hdf5_file_path, 'r') as f:
            date = f.attrs['date']
            retinotopy_altitude = f['retinotopy_altitude'][:]
            retinotopy_azimuth = f['retinotopy_azimuth'][:]
            visual_sign = f['visual_sign'][:]
            vasculature_image = f['vasculature_image'][:]
            defocus_image = f['defocus_image'][:]
            visual_sign_pixel_size = f['visual_sign'].attrs['pixel_size_x_um']
            vasculature_pixel_size = f['vasculature_image'].attrs['pixel_size_x_um']
            retinotopy_altitude_shape = f['retinotopy_altitude'].shape
            retinotopy_azimuth_shape = f['retinotopy_azimuth'].shape
            visual_sign_shape = f['visual_sign'].shape
            vasculature_image_shape = f['vasculature_image'].shape
            defocus_image_shape = f['defocus_image'].shape
        label_map_image = imageio.v2.imread(label_map_path) if label_map_path is not None else None
        return cls(
            date,
            retinotopy_altitude,
            retinotopy_azimuth,
            visual_sign,
            vasculature_image,
            defocus_image,
            visual_sign_pixel_size,
            vasculature_pixel_size,
            retinotopy_altitude_shape,
            retinotopy_azimuth_shape,
            visual_sign_shape,
            vasculature_image_shape,
            defocus_image_shape,
            label_map_image
        )

class ISIMetricsAndVisualization:
    """Provides metrics calculation and visualization methods for ISI data using a data loader."""
    def __init__(self, data_loader):
        """Initialize with a loaded ISIDataLoader instance and set up image dimensions."""
        self.data = data_loader
        self.label_map_image = data_loader.label_map_image
        # Set up image dimensions
        self.vasculature_image_width = self.data.vasculature_image_shape[1]
        self.vasculature_image_height = self.data.vasculature_image_shape[0]
        self.visual_sign_width = self.data.visual_sign_shape[1]
        self.visual_sign_height = self.data.visual_sign_shape[0]
        self.retinotopy_altitude_width = self.data.retinotopy_altitude_shape[1]
        self.retinotopy_altitude_height = self.data.retinotopy_altitude_shape[0]
        self.retinotopy_azimuth_width = self.data.retinotopy_azimuth_shape[1]
        self.retinotopy_azimuth_height = self.data.retinotopy_azimuth_shape[0]
        self.defocus_image_width = self.data.defocus_image_shape[1]
        self.defocus_image_height = self.data.defocus_image_shape[0]


    def generate_qc_metric_and_images(self, eccentricity_retinotopic_zero_path, eccentricity_v_one_centroid_path, target_map_path):
        """Generate QC metrics and images for ISI segmentation and targeting using only the label_map_image."""
        if self.label_map_image is None:
            raise Exception("label_map_image must be provided to ISIDataLoader for region extraction.")

        # Build annotated_regions_data from label_map_image only, using full-size masks
        annotated_regions_data = []
        label_map = self.label_map_image
        structure_ids = np.unique(label_map)
        print(f"Found {len(structure_ids)} unique structure IDs in label map.")
        for structure_id in structure_ids:
            if structure_id == 0:
                continue  # skip background
            mask = (label_map == structure_id).astype(np.uint8)
            annotated_regions_data.append({
                'structure_id': int(structure_id),
                'region_mask': mask
            })
            

        # Use self.data for all data access
        altitude_phase = self.data.retinotopy_altitude
        azimuth_phase = self.data.retinotopy_azimuth
        vasculature = self.data.vasculature_image.astype(float)
        sign = self.data.visual_sign
        map_pixel_size = self.data.visual_sign_pixel_size
        vasculature_pixel_size = self.data.vasculature_pixel_size

        altitude_scale = 0.322
        azimuth_scale = 0.383 
        altitude = altitude_phase * altitude_scale
        azimuth = azimuth_phase * azimuth_scale

        eccentricity_ret_zero = np.degrees(eccentricity(azimuth, altitude, 0.0, 0.0))
        
        VISp_blob = None
        outlines = []

        for annotated_region in annotated_regions_data:
            structure_id = annotated_region['structure_id']
            region_mask = annotated_region['region_mask']

            region_data = {}
            region_data['region_mask'] = region_mask

            # compute centroid
            centroid = [np.mean(x) for x in np.where(region_mask)]
            region_data['y_centroid'] = centroid[0]
            region_data['x_centroid'] = centroid[1]

            # compute azimuth/altitude min, max, range, and bias
            (region_data['azimuth_min'],region_data['azimuth_max'],region_data['azimuth_range'],region_data['azimuth_bias']) = retinotopy_metric(region_mask, azimuth)
            (region_data['altitude_min'],region_data['altitude_max'],region_data['altitude_range'],region_data['altitude_bias']) = retinotopy_metric(region_mask, altitude)

            # eccentricity at centroid
            region_data['eccentricity_at_centroid'] = eccentricity_ret_zero[round(region_data['y_centroid']), round(region_data['x_centroid'])]

            outlines.append(outline_mask(region_mask))

            if structure_id == 1:
                VISp_blob = region_data
        
        if VISp_blob is None:
            raise Exception("Expected to find VISp region with structure id of 385 but did not find one")

        azimuth_target = azimuth[round(VISp_blob['y_centroid']), round(VISp_blob['x_centroid'])]
        altitude_target = altitude[round(VISp_blob['y_centroid']), round(VISp_blob['x_centroid'])]
        eccentricity_V1_centroid = np.degrees(eccentricity(azimuth, altitude, azimuth_target, altitude_target ))

        #
        # Create higher resolution vasculature map background image
        data = window_level(vasculature, np.min(vasculature), np.max(vasculature))
        background_im = Image.fromarray(np.uint8(plt.cm.gray(data)*255))
        
        # ----------------------------
        # This creates eccentricity retintopic zero map
        # ----------------------------
            
        # foreground - eccentricity map  
        data = window_level(eccentricity_ret_zero, 0, 50)    
        foreground_im = Image.fromarray(np.uint8(plt.cm.hsv(data)*255))
        
        # resize foreground image to background
        foreground_im = foreground_im.resize( background_im.size )
            
        # alpha blend images    
        composite_im = Image.blend( foreground_im, background_im, 0.65 )
        print("saving " + eccentricity_retinotopic_zero_path)
        composite_im.save(eccentricity_retinotopic_zero_path)
        
        # --------------------------
        # This creates eccentricity from V1 centroid map and targeting map
        # ---------------------------
        data = window_level(eccentricity_V1_centroid, 0, 50)    
        foreground_im = Image.fromarray(np.uint8(plt.cm.hsv(data)*255))
        
        # resize foreground image to background
        foreground_im = foreground_im.resize( background_im.size )
            
        # alpha blend images    
        composite_im = Image.blend( foreground_im, background_im, 0.65 )
        print("saving " + eccentricity_v_one_centroid_path)
        composite_im.save(eccentricity_v_one_centroid_path)  

        # erosion structure element
        structure = ndimage.generate_binary_structure(2, 1)
        structure = ndimage.iterate_structure( structure, 3 )
        
        line_alpha = 100.0

        target_mask_ten = eccentricity_V1_centroid < 10.0
        target_mask_five = eccentricity_V1_centroid < 5.0
        target_mask_one = eccentricity_V1_centroid < 1.0

        target_mask_ten = outline_mask(target_mask_ten)
        target_mask_five = outline_mask(target_mask_five)
        target_mask_one = outline_mask(target_mask_one)
        
        # resize mask
        outline_im_ten = Image.fromarray( np.uint8( target_mask_ten * line_alpha )  )
        outline_im_ten = outline_im_ten.resize( background_im.size )

        outline_im_five = Image.fromarray( np.uint8( target_mask_five * line_alpha )  )
        outline_im_five = outline_im_five.resize( background_im.size )

        outline_im_one = Image.fromarray( np.uint8( target_mask_one * line_alpha )  )
        outline_im_one = outline_im_one.resize( background_im.size )

        red_im = Image.fromarray( np.uint8(plt.cm.Reds_r(np.zeros(vasculature.shape)) * 255 ) )
        green_im = Image.fromarray( np.uint8(plt.cm.Greens_r(np.zeros(vasculature.shape)) * 255 ) )
        
        # erode mask
        arr_ten = np.asarray( outline_im_ten, dtype=np.uint8 )
        arr_ten = ndimage.binary_erosion(arr_ten,structure).astype(arr_ten.dtype)
        outline_im_ten = Image.fromarray( np.uint8(arr_ten * line_alpha) )

        arr_five = np.asarray( outline_im_five, dtype=np.uint8 )
        arr_five = ndimage.binary_erosion(arr_five,structure).astype(arr_five.dtype)
        outline_im_five = Image.fromarray( np.uint8(arr_five * line_alpha) )

        arr_one = np.asarray( outline_im_one, dtype=np.uint8 )
        arr_one = ndimage.binary_erosion(arr_one,structure).astype(arr_one.dtype)
        outline_im_one = Image.fromarray( np.uint8(arr_one * line_alpha) )
        
        target_im = Image.composite( foreground_im, background_im, outline_im_ten )
        target_im = Image.composite( green_im, target_im, outline_im_five )
        target_im = Image.composite( red_im, target_im, outline_im_one )
        
        # generate segmentation outline image
        region_outlines = np.zeros( altitude.shape )
        # for idx, r in enumerate(output_json['annotated_regions']) :
            # region_outlines[r['outline_mask'] > 0] = 1.0

        for outline in outlines:
            region_outlines[outline > 0] = 1.0

            
        # resize mask
        outline_im = Image.fromarray( np.uint8(region_outlines * line_alpha ) )
        outline_im = outline_im.resize( background_im.size )
        
        # erode mask
        arr = np.asarray( outline_im, dtype=np.uint8 )
        arr = ndimage.binary_erosion(arr, structure).astype(arr.dtype) 
        outline_im = Image.fromarray( np.uint8(arr * line_alpha) )
        
        # create a blue image
        blue_im = Image.fromarray( np.uint8(plt.cm.Blues_r(np.zeros(vasculature.shape)) * 255 ) )
        
        # composite the region outline with the target image
        target_im = Image.composite( blue_im, target_im, outline_im )
        print("saving " + target_map_path)
        target_im.save(target_map_path)

    def create_visual_sign_image(self, sign_map_path):
        """Create and save a visual sign image using a colormap and alpha mask."""
        arr = self.data.visual_sign
        visual_sign_shape = arr.shape
        visual_sign_width = visual_sign_shape[1]
        visual_sign_height = visual_sign_shape[0]
        alpha = 85.0
        threshold = 0.25
        mask = np.uint8((abs(arr) > threshold) * alpha)
        mask_im = Image.fromarray(mask)
        arr = (arr + 1.0) / 2.0
        sign_im = Image.fromarray(np.uint8(plt.cm.jet(arr) * 255))
        print("saving " + sign_map_path)
        sign_im.save(sign_map_path)

    def create_retinotopy_altitude_image(self, retinotopy_vertical_path):
        """Create and save a retinotopy altitude image from the input file."""
        arr = self.data.retinotopy_altitude
        arr = (arr + math.pi)/(2.0 * math.pi)
        im = Image.fromarray(np.uint8(plt.cm.hsv(arr)*255))
        print("saving " + retinotopy_vertical_path)
        im.save(retinotopy_vertical_path)

    def create_retinotopy_azimuth_image(self, retinotopy_horizontal_path):
        """Create and save a retinotopy azimuth image from the input file."""
        arr = self.data.retinotopy_azimuth
        arr = (arr + math.pi)/(2.0 * math.pi)
        im = Image.fromarray(np.uint8(plt.cm.hsv(arr)*255))
        print("saving " + retinotopy_horizontal_path)
        im.save(retinotopy_horizontal_path)

    def create_vasculature_image(self, vasculature_path):
        """Create and save a vasculature image from the input file."""
        arr = self.data.vasculature_image.astype(float)
        arr = arr/arr.max()
        im = Image.fromarray(np.uint8(plt.cm.gray(arr)*255))
        print("saving " + vasculature_path)
        im.save(vasculature_path)
        png_vasculature_path = os.path.splitext(vasculature_path)[0]+'.png'
        im.save(png_vasculature_path)

    def create_defocus_image(self, isi_imaging_plane_path):
        """Create and save a defocus image from the input file."""
        arr = self.data.defocus_image.astype(float)
        arr = arr/arr.max()
        im = Image.fromarray(np.uint8(plt.cm.gray(arr)*255))
        print("saving " + isi_imaging_plane_path)
        im.save(isi_imaging_plane_path)

if __name__ == "__main__":
    """Main function to test ISIDataLoader and ISIMetricsAndVisualization on sample data."""
    import glob
    sample_files = glob.glob("/data/isi_segmentation_model/*_processed.hdf5")
    label_map_files = glob.glob("/data/isi_segmentation_model/*_label_map.png")
    if not sample_files:
        print("No sample HDF5 files found in /data/isi_segmentation_model/")
        sys.exit(1)
    if not label_map_files:
        print("No label map PNG files found in /data/isi_segmentation_model/")
        sys.exit(1)
    hdf5_file = sample_files[0]
    label_map_path = label_map_files[0]
    print(f"Using sample file: {hdf5_file}")
    print(f"Using label map: {label_map_path}")

    # Instantiate loader and metrics/visualization
    data = ISIData.from_files(hdf5_file, label_map_path=label_map_path)
    metrics = ISIMetricsAndVisualization(data)

    # Example output paths (these would be replaced with real ones in production)
    output_dir_path = "/results/qc_images"
    os.makedirs(output_dir_path, exist_ok=True)
    sign_map_path = os.path.join(output_dir_path, "sign_map.png")
    retinotopy_vertical_path = os.path.join(output_dir_path, "retinotopy_vertical.png")
    retinotopy_horizontal_path = os.path.join(output_dir_path, "retinotopy_horizontal.png")
    vasculature_path = os.path.join(output_dir_path, "vasculature.png")
    isi_imaging_plane_path = os.path.join(output_dir_path, "defocus.png")
    isi_overlay_path = os.path.join(output_dir_path, "overlay.png")
    eccentricity_retinotopic_zero_path = os.path.join(output_dir_path, "eccentricity_retinotopic_zero.png")
    eccentricity_v_one_centroid_path = os.path.join(output_dir_path, "eccentricity_v_one_centroid.png")
    target_map_path = os.path.join(output_dir_path, "target_map.png")

    # Call representative methods
    metrics.create_visual_sign_image(sign_map_path)
    metrics.create_retinotopy_altitude_image(retinotopy_vertical_path)
    metrics.create_retinotopy_azimuth_image(retinotopy_horizontal_path)
    metrics.create_vasculature_image(vasculature_path)
    metrics.create_defocus_image(isi_imaging_plane_path)

    # Try to run QC metrics and target map generation
    try:
        metrics.generate_qc_metric_and_images(
            eccentricity_retinotopic_zero_path,
            eccentricity_v_one_centroid_path,
            target_map_path
        )
        print("QC metrics and target map generated successfully.")
    except Exception as e:
        print(f"Error running generate_qc_metric_and_images: {e}")

    print("Test run complete.")