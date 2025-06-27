#!/usr/bin/python
import h5py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import xml.etree.cElementTree as ET
import json
from scipy import ndimage
from extract_target_image import ExtractTargetImage

class IsiExperimentExtraction:
  def __init__(self, argv):

    if len(argv) != 12:
      raise Exception("Expected 12 arguments but received " + str(len(argv)))

    retinotopy_vertical = argv[0]
    retinotopy_horizontal = argv[1]
    sign_map = argv[2]
    vasculature = argv[3]
    isi_imaging_plane = argv[4]
    isi_overlay = argv[5]
    hdf5_file = argv[6]
    module_output_file = argv[7]
    input_json = argv[8]
    eccentricity_retinotopic_zero = argv[9]
    eccentricity_v_one_centroid = argv[10]
    target_map = argv[11]

    print 'hdf5_file: ' + hdf5_file

    data_input_file = h5py.File(hdf5_file, 'r')

    date = data_input_file.attrs['date']

    self.create_visual_sign_image(data_input_file, sign_map)
    self.create_retinotopy_altitude_image(data_input_file, retinotopy_vertical)  
    self.create_retinotopy_azimuth_image(data_input_file, retinotopy_horizontal)  
    self.create_vasculature_image(data_input_file, vasculature)
    self.create_isi_overlay_image(data_input_file, isi_overlay)
    self.create_defocus_image(data_input_file, isi_imaging_plane)  
    self.create_annotated_regions_image(input_json)  

    ExtractTargetImage().generate_qc_metric_and_images(data_input_file, eccentricity_retinotopic_zero, eccentricity_v_one_centroid, target_map, input_json)

    self.create_module_output_file(module_output_file, date)

    data_input_file.close()

  def create_module_output_file(self, module_output_file, date):
    #write width and height to xml
    isi_experiment = ET.Element("isi_experiment")
    images = ET.SubElement(isi_experiment, "images")

    ET.SubElement(images, "date_of_acquisition").text = str(date)

    ET.SubElement(images, "visual_sign_width").text = str(self.visual_sign_width)
    ET.SubElement(images, "visual_sign_height").text = str(self.visual_sign_height)

    ET.SubElement(images, "retinotopy_altitude_width").text = str(self.retinotopy_altitude_width)
    ET.SubElement(images, "retinotopy_altitude_height").text = str(self.retinotopy_altitude_height)

    ET.SubElement(images, "retinotopy_azimuth_width").text = str(self.retinotopy_azimuth_width)
    ET.SubElement(images, "retinotopy_azimuth_height").text = str(self.retinotopy_azimuth_height)

    ET.SubElement(images, "vasculature_image_width").text = str(self.vasculature_image_width)
    ET.SubElement(images, "vasculature_image_height").text = str(self.vasculature_image_height)

    ET.SubElement(images, "defocus_image_width").text = str(self.defocus_image_width)
    ET.SubElement(images, "defocus_image_height").text = str(self.defocus_image_height)

    ET.SubElement(images, "overlay_width").text = str(self.overlay_width)
    ET.SubElement(images, "overlay_height").text = str(self.overlay_height)

    ET.SubElement(images, "target_map_width").text = str(self.vasculature_image_width)
    ET.SubElement(images, "target_map_height").text = str(self.vasculature_image_height)

    ET.SubElement(images, "eccentricity_width").text = str(self.vasculature_image_width)
    ET.SubElement(images, "eccentricity_height").text = str(self.vasculature_image_height)

    tree = ET.ElementTree(isi_experiment)

    print "saving " + module_output_file
    tree.write(module_output_file)

  def eccentricity(self, az, alt, az_center, alt_center ):
    daz = az - az_center
    dalt = alt - alt_center
    ecc = np.arctan( np.sqrt( np.square(np.tan(dalt)) +
                              np.square(np.tan(daz))/np.square(np.cos(dalt)))
                   )
    return ecc
        
  def retinotopy_metric(self, mask, map ) :
    ind = np.where( mask > 0 )
    vals = map[ind]
    maxv = np.degrees(np.max(vals))
    minv = np.degrees(np.min(vals))
    return( minv, maxv, maxv - minv, abs( minv + maxv ) )
        
  def window_level(self, input, cmin, cmax ) :
    data = np.copy(input)
    crange = cmax - cmin
    data[data < cmin] = cmin
    data[data > cmax] = cmax
    data -= cmin
    data /= (cmax-cmin)
    return data
        
  def outline_mask(self, mask ) :
    edge_horizont = ndimage.sobel(mask, 0)
    edge_vertical = ndimage.sobel(mask, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    outline = np.zeros( mask.shape )
    outline[magnitude > 0] = 1
    outline = np.uint8(outline)
    return outline

  def create_annotated_regions_image(self, input_json):
    try:
      with open(input_json) as data_file:    
        data = json.load(data_file)
    except:
      raise Exception("Unexpected error:", sys.exc_info()[0])

    retinotopy_altitude_shape = self.altitude_dataset.shape
    annotated_region_image_width = retinotopy_altitude_shape[1]
    annotated_region_image_height = retinotopy_altitude_shape[0]
    annotated_regions_output_file = data['annotated_regions_output_file']
    self.annotated_regions_data = data['annotated_regions']

    #initialize whole image to black 
    annotated_regions = np.zeros((annotated_region_image_height,annotated_region_image_width, 4), dtype = np.uint8)

    #loop through each annotated region and draw in each annotated region
    for annotated_region in self.annotated_regions_data:
      annotated_region_width = annotated_region['width']
      annotated_region_height = annotated_region['height']
      offset_x = annotated_region['x']
      offset_y = annotated_region['y']
      structure_id = annotated_region['structure_id']
      mask_matrix = annotated_region['mask_matrix']

      annotated_region_array = mask_matrix

      index_to_red = 0
      index_to_green = 1
      index_to_blue = 2
      index_to_alpha = 3

      color = structure_id

      #if we get a structure of 0 this means the sructure is not set. Set color for this to be max value (white), min value (black) in image means pixel is not in an annotated region
      if color == 0:
        alpha = 255
        red = 255
        green = 255
        blue = 255
      else:

        #extract the 8 bit colors from the 32bit structure_id (color)
        blue = color & 0xFF
        color = color >> 8
        green = color & 0xFF
        color = color >> 8
        red = color & 0xFF
        color = color >> 8
        alpha = color & 0xFF

      #loop through annotated region and set pixels colors
      for y in range(0, annotated_region_height):
        for x in range(0, annotated_region_width):
          if annotated_region_array[y][x]:       
            annotated_regions[y + offset_y][x + offset_x][index_to_alpha] = alpha
            annotated_regions[y + offset_y][x + offset_x][index_to_red] = red
            annotated_regions[y + offset_y][x + offset_x][index_to_green] = green
            annotated_regions[y + offset_y][x + offset_x][index_to_blue] = blue
    
    im = Image.fromarray(annotated_regions)

    print "saving " + annotated_regions_output_file
    im.save(annotated_regions_output_file)

  def create_defocus_image(self, data_input_file, isi_imaging_plane):
    #
    # process defocus_image
    #
    dataset = data_input_file['defocus_image']
    arr = dataset[()] # read scalar dataset as nump arrary

    defocus_image_shape = dataset.shape
    self.defocus_image_width = defocus_image_shape[1]
    self.defocus_image_height = defocus_image_shape[0]

    # for vasculature scale to [0,max]
    arr = arr.astype(float)
    arr = arr/arr.max()

    # apply gray colormap
    im = Image.fromarray(np.uint8(plt.cm.gray(arr)*255))

    print "saving " + isi_imaging_plane
    im.save(isi_imaging_plane)

  def create_isi_overlay_image(self, data_input_file, isi_overlay):
    # window level [0.1,1]
    self.vas_arr = (self.vas_arr-0.1)/0.9

    # apply gray colormap
    self.vasculature_im = Image.fromarray(np.uint8(plt.cm.gray(self.vas_arr)*255))

    # resize to visual sign map
    self.vasculature_im = self.vasculature_im.resize(self.sign_im.size)

    #
    # composite
    #

    composite_im = Image.composite(self.sign_im,self.vasculature_im, self.mask_im)

    overlay_size = self.sign_im.size
    self.overlay_width = overlay_size[0]
    self.overlay_height = overlay_size[1]

    print "saving " + isi_overlay
    composite_im.save(isi_overlay)
    #Then create a png version of the image for inclusion in PowerBI reports
    png_overlay = os.path.splitext(isi_overlay)[0]+'.png'
    composite_im.save(png_overlay)

  def create_vasculature_image(self, data_input_file, vasculature):
    #
    # process vasculature_image
    #
    dataset = data_input_file['vasculature_image']
    self.vas_arr = dataset[()] # read scalar dataset as nump arrary

    vasculature_image_shape = dataset.shape
    self.vasculature_image_width = vasculature_image_shape[1]
    self.vasculature_image_height = vasculature_image_shape[0]

    # for vasculature scale to [0,max]
    self.vas_arr = self.vas_arr.astype(float)
    self.vas_arr = self.vas_arr/self.vas_arr.max()

    # apply gray colormap
    im = Image.fromarray(np.uint8(plt.cm.gray(self.vas_arr)*255))

    print "saving " + vasculature
    im.save(vasculature)
    #Then create a png version of the image for inclusion in PowerBI reports
    png_vasculature = os.path.splitext(vasculature)[0]+'.png'
    im.save(png_vasculature)

  def create_retinotopy_azimuth_image(self, data_input_file, retinotopy_horizontal):
    #
    # process retinotopy_azimuth
    #
    self.azimuth_dataset = data_input_file['retinotopy_azimuth']
    arr = self.azimuth_dataset[()] # read scalar dataset as nump arrary

    retinotopy_azimuth_shape = self.azimuth_dataset.shape
    self.retinotopy_azimuth_width = retinotopy_azimuth_shape[1]
    self.retinotopy_azimuth_height = retinotopy_azimuth_shape[0]

    # for retinotopy scale to [-pi,pi]
    arr = (arr + math.pi)/(2.0 * math.pi)

    # apply jet colormap
    im = Image.fromarray(np.uint8(plt.cm.hsv(arr)*255))

    print "saving " + retinotopy_horizontal
    im.save(retinotopy_horizontal)


  def create_retinotopy_altitude_image(self, data_input_file, retinotopy_vertical):
    #
    # process retinotopy_altitude
    #
    self.altitude_dataset = data_input_file['retinotopy_altitude']
    arr = self.altitude_dataset[()] # read scalar dataset as nump arrary

    retinotopy_altitude_shape = self.altitude_dataset.shape
    self.retinotopy_altitude_width = retinotopy_altitude_shape[1]
    self.retinotopy_altitude_height = retinotopy_altitude_shape[0]

    # for retinotopy scale [-pi,pi]
    arr = (arr + math.pi)/(2.0 * math.pi)

    # apply jet colormap
    im = Image.fromarray(np.uint8(plt.cm.hsv(arr)*255))

    print "saving " + retinotopy_vertical
    im.save(retinotopy_vertical)

  def create_visual_sign_image(self, data_input_file, sign_map):
    # process visual sign
    dataset = data_input_file['visual_sign']
    arr = dataset[()] # read scalar dataset as nump arrary

    #get width and height
    visual_sign_shape = dataset.shape
    self.visual_sign_width = visual_sign_shape[1]
    self.visual_sign_height = visual_sign_shape[0]

    #
    # create mask by thresholding low values of signed map
    #
    alpha = 85.0
    threshold = 0.25
    mask = np.uint8( ( abs(arr) > threshold ) * alpha )
    self.mask_im = Image.fromarray(mask)

    # we want to map jet scaled to [-1,1]
    arr = (arr + 1.0)/2.0

    # apply jet colormap
    self.sign_im = Image.fromarray(np.uint8(plt.cm.jet(arr)*255))

    print "saving " + sign_map
    self.sign_im.save(sign_map)

if __name__ == "__main__":
  print 'Running python isi extraction'
  IsiExperimentExtraction(sys.argv[1:])
  print 'Finished python code'