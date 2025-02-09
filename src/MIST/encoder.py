'''MIST Encoder'''

import pandas as pd
import cv2
import os
import multiprocessing
import numpy as np
from functools import partial
import pydicom
import nibabel as nib
from tqdm.contrib.concurrent import process_map
import openjphpy as ojph
from openjphpy.backend import Tileparts
from .utils import *

#### Encoder 2D

def __encode2d_worker(
  path,
  input_dir, 
  output_dir
):
  try:
    input_path = os.path.join(input_dir, path)
    # Check for valid format and dimension
    if path.lower().endswith(VALID_FORMATS_2D[:3]):
      img = cv2.imread(input_path, 0)
    elif path.lower().endswith(VALID_FORMATS_2D[3]):
      ds = pydicom.dcmread(input_path)
      if ds.get('NumberOfFrames'):
        raise ValueError('Invalid format! Data is not 2D')
      img = ds.pixel_array
    elif path.lower().endswith(VALID_FORMATS_2D[4]):
      raise NotImplementedError('NRRD data not yet supported')
    elif path.lower().endswith(VALID_FORMATS_2D[5:]):
      ds = nib.load(input_path)
      if ds.header['dim'][0] != 2:
        raise ValueError('Invalid format! Data is not 2D')
      img = ds.get_fdata()
    else:
      raise ValueError(f'Invalid format! Acceptable formats: {VALID_FORMATS_2D}')
    scl_intercept, scl_slope = calculate_rescale_attributes(img)
    img = encoder_apply_rescale_attributes(img, scl_intercept, scl_slope)
    # Calculate number of decompositon levels
    num_decomps = calculate_num_decomps(img.shape)
    output_path = os.path.join(
      output_dir, 
      'htj2k', 
      strip_extension(path) + '.j2c'
    )
    # Encode image as HTJ2K
    encode_time = ojph.encode(
      output_path, 
      img, 
      reversible = True,
      num_decomps = num_decomps,
      tlm_marker = True,
      tileparts = Tileparts.R
    )
    # Return encoding metadata
    x, y = img.shape
    markers = get_scan_markers(output_path)
    return np.array([path, x, y, num_decomps, scl_intercept, scl_slope, encode_time, markers], dtype=object)
  except:
    return np.array([path, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ''], dtype=object)

def encode2d(
  paths,
  input_dir,
  output_dir,
  use_multiprocessing = True,
  max_workers = -1
):
  os.makedirs(output_dir, exist_ok=True)
  if isinstance(paths, str):
    paths = [paths]
  if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
    raise ValueError('Invalid type! For single-image encoding, `paths` must be a string and for multi-image encoding, `paths` must be a list of strings')
  # Use multiprocessing to encode images if enabled
  if use_multiprocessing:
    # Ensure `max_workers` value is valid
    if not max_workers > 0:
      # Default to max processes if value is -1
      if max_workers == -1:
        max_workers = multiprocessing.cpu_count()
      else:
        raise ValueError('Invalid value! `max_workers` must be either greater than 0 to limit number of processes utilized or set to -1 to use max number of processes')
    else:
      # Default to max processes if `max_workers` exceeds CPU limit
      if max_workers > multiprocessing.cpu_count():
        max_workers = multiprocessing.cpu_count()
        print(f'`max_workers` exceeds max number of processes. Defaulting value to -1')
    print(f'Using {max_workers} worker(s) to encode')
    encode_metadata = process_map(partial(__encode2d_worker, input_dir=input_dir, output_dir=output_dir), paths, max_workers=max_workers, chunksize=1)
  else:
    encode_metadata = []
    for path in paths:
      encode_metadata += [__encode2d_worker(path, input_dir, output_dir)]
  # Save encoding metadata
  pd.DataFrame(
    np.array(encode_metadata, dtype=object),
    columns = ['image', 'x', 'y', 'num_decomps', 'scl_intercept', 'scl_slope', 'encode_time', 'markers']
  ).sort_values('image').to_csv(os.path.join(output_dir, 'encode_metadata.csv'), index=False)

#### Encoder 3D

def __encode3d_worker(
  path,
  input_dir, 
  output_dir
):
  if isinstance(path, str):
    input_path = os.path.join(input_dir, path)
    # TODO: DICOM 3D is (N,X,Y) while NifTI is (X,Y,N)
    # TODO: Support for grayscale and RGB data
    if path.lower().endswith(VALID_FORMATS_3D[0]):
      raise NotImplementedError('DICOM data not yet supported')
    elif path.lower().endswith(VALID_FORMATS_3D[1]):
      raise NotImplementedError('NRRD data not yet supported')
    elif path.lower().endswith(VALID_FORMATS_3D[2:]):
        ds = nib.load(input_path)
        if ds.header['dim'][0] != 3:
          raise ValueError('Invalid format! Data is not 3D')
        img = ds.get_fdata()
    else:
      raise ValueError(f'Invalid format! Acceptable formats: {VALID_FORMATS_3D}')
    scl_intercept, scl_slope = calculate_rescale_attributes(img)
    metadata = []
    x, y = img.shape[:2]
    slices = img.shape[-1]
    for slice in range(slices):
      img_slice = img[:,:,slice]
      img_slice = encoder_apply_rescale_attributes(img_slice, scl_intercept, scl_slope)
      # Calculate number of decompositon levels
      num_decomps = calculate_num_decomps(img_slice.shape)
      output_path = os.path.join(
        output_dir, 
        'htj2k', 
        strip_extension(path),
        f'{slice+1}.j2c'
      )
      # Encode image as HTJ2K
      encode_time = ojph.encode(
        output_path, 
        img_slice, 
        reversible = True,
        num_decomps = num_decomps,
        tlm_marker = True,
        tileparts = Tileparts.R
      )
      # Return encoding metadata
      markers = get_scan_markers(output_path)
      metadata += [[path, slice+1, slices, x, y, num_decomps, scl_intercept, scl_slope, encode_time, markers]]
    return np.array(metadata, dtype=object)
  else:
    # TODO: Implement 3D data from 2D DICOM slices
    raise NotImplementedError('3D DICOM data not yet supported')

def encode3d(
  paths,
  input_dir,
  output_dir,
  use_multiprocessing = True,
  max_workers = -1,
):
  if isinstance(paths, str):
    paths = [paths]
  if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths) or not all(all(isinstance(pp, str) for pp in p) for p in paths):
    raise ValueError('Invalid type! For single-volume encoding, `paths` must be a string path for 3D imaging data or a list of string paths for sequence of 2D slices. For multi-volume encoding, `paths` must be a list of string paths for 3D imaging data or a list of list of string paths for sequences of 2D slices.')
  # Use multiprocessing to encode images if enabled
  if use_multiprocessing:
    # Ensure `max_workers` value is valid
    if not max_workers > 0:
      # Default to max processes if value is -1
      if max_workers == -1:
        max_workers = multiprocessing.cpu_count()
      else:
        raise ValueError('Invalid value! `max_workers` must be either greater than 0 to limit number of processes utilized or set to -1 to use max number of processes')
    else:
      # Default to max processes if `max_workers` exceeds CPU limit
      if max_workers > multiprocessing.cpu_count():
        max_workers = multiprocessing.cpu_count()
        print(f'`max_workers` exceeds max number of processes. Defaulting value to -1')
    print(f'Using {max_workers} worker(s) to encode')
    encode_metadata = process_map(partial(__encode3d_worker, input_dir=input_dir, output_dir=output_dir), paths, max_workers=max_workers, chunksize=1)
  else:
    encode_metadata = []
    for path in paths:
      encode_metadata += [__encode3d_worker(path, input_dir, output_dir)]
  encode_metadata = np.concatenate(encode_metadata, axis=0)
  # Save encoding metadata
  pd.DataFrame(
    encode_metadata,
    columns = ['image', 'slice', 'total_slices', 'x', 'y', 'num_decomps', 'scl_intercept', 'scl_slope', 'encode_time', 'markers']
  ).sort_values(['image', 'slice']).to_csv(os.path.join(output_dir, 'encode_metadata.csv'), index=False)