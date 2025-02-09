'''MIST Decoder'''

import pandas as pd
import os
import multiprocessing
import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map
import openjphpy as ojph
from .utils import *

#### Decoder 2D

def __decode2d_worker(
  data,
  decomp_level,
  input_dir,
  output_dir
):
  # Read metadata
  # path, num_decomps, rescale_intensity, min_intensity, max_intensity = data
  path, num_decomps, scl_intercept, scl_slope = data
  if decomp_level != None and decomp_level < 0:
    raise ValueError(f'Invalid scan to decode. Value must be greater than 0 but less than num_decomps: {num_decomps}')
  input_path = os.path.join(
    input_dir, 
    'htj2k', 
    strip_extension(path) + '.j2c'
  )
  # Decode HTJ2K at specified decomposition level
  # For N decompositions, the N+1 decomposition levels exist
  if decomp_level == None or decomp_level > num_decomps:
    # Sanity check! Default to 0 i.e., full resolution
    img, decode_time = ojph.decode(input_path, skip_res=0)
  else:
    img, decode_time = ojph.decode(input_path, skip_res=int(num_decomps-decomp_level))
  # Rescale pixel values to original range
  # if rescale_intensity:
  #   img = rescale_intensities(img, new_range=(min_intensity, max_intensity))
  img = decoder_apply_rescale_attributes(img, scl_intercept, scl_slope)
  # Save npz (compressed) file 
  if decomp_level == None:
    output_path = os.path.join(
      output_dir, 
      f'npy_scan_{num_decomps+1}',
      strip_extension(path) + '.npz'
    )
  else:
    output_path = os.path.join(
      output_dir, 
      f'npy_scan_{decomp_level+1}',
      strip_extension(path) + '.npz'
    )
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  np.savez_compressed(output_path, img)
  # Return decoding metadata
  return [path, decode_time]

def decode2d(
  paths,
  decomp_level,
  input_dir,
  output_dir = None,
  use_multiprocessing = True,
  max_workers = -1
):
  if isinstance(paths, str):
    paths = [paths]
  if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
    raise ValueError('Invalid type! For single-image decoding, `paths` must be a string and for multi-image decoding, `paths` must be a list of strings')
  # Default output directory to subdirectory in input directory
  if not output_dir:
    output_dir = input_dir
  else:
    os.makedirs(output_dir, exist_ok=True)
  encode_df = pd.read_csv(os.path.join(input_dir, 'encode_metadata.csv'))
  if decomp_level == None:
    decomp_level = encode_df['num_decomps'].max()
  # if decomp_level > encode_df['num_decomps'].max():
  #   raise ValueError(f'Invalid scan to decode. Value must be greater than 0 but less than num_decomps: {encode_df["num_decomps"].max()}')
  # encode_df = encode_df[encode_df['image'].isin(paths)][['image', 'num_decomps', 'rescale_intensity', 'min_intensity', 'max_intensity']]
  encode_df = encode_df[encode_df['image'].isin(paths)][['image', 'num_decomps', 'scl_intercept', 'scl_slope']]
  data = list(encode_df.itertuples(index=False, name=None))
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
    print(f'Using {max_workers} worker(s) to decode')
    decode_metadata = process_map(partial(__decode2d_worker, decomp_level=decomp_level, input_dir=input_dir, output_dir=output_dir), data, max_workers=max_workers, chunksize=1)
  else:
    decode_metadata = []
    for d in data:
      decode_metadata += [__decode2d_worker(d, decomp_level, input_dir, output_dir)]
  # Save decoding metadata
  # if decomp_level == None:
  #   pd.DataFrame(
  #     np.array(decode_metadata, dtype=object),
  #     columns = ['image', 'decode_time']
  #   ).sort_values('image').to_csv(os.path.join(output_dir, 'decode_full_metadata.csv'), index=False)
  # else:
  #   pd.DataFrame(
  #     np.array(decode_metadata, dtype=object),
  #     columns = ['image', 'decode_time']
  #   ).sort_values('image').to_csv(os.path.join(output_dir, f'decode_{decomp_level+1}_metadata.csv'), index=False)
  pd.DataFrame(
    np.array(decode_metadata, dtype=object),
    columns = ['image', 'decode_time']
  ).sort_values('image').to_csv(os.path.join(output_dir, f'decode_{decomp_level+1}_metadata.csv'), index=False)

#### Decoder 3D

def __decode3d_worker(
  data,
  decomp_level,
  input_dir,
  output_dir
):
  # Read metadata
  path, total_slices, num_decomps, scl_intercept, scl_slope = data
  if decomp_level != None and decomp_level < 0:
    raise ValueError(f'Invalid scan to decode. Value must be greater than 0 but less than num_decomps: {num_decomps}')
  decode_time = 0
  img = []
  for slice in range(total_slices):
    input_path = os.path.join(
      input_dir, 
      'htj2k', 
      strip_extension(path),
      f'{slice+1}.j2c'
    )
    # Decode HTJ2K at specified decomposition level
    # For N decompositions, the N+1 decomposition levels exist
    if decomp_level == None or decomp_level > num_decomps:
      # Sanity check! Default to 0 i.e., full resolution
      img_slice, decode_time_slice = ojph.decode(input_path, skip_res=0)
    else:
      img_slice, decode_time_slice = ojph.decode(input_path, skip_res=num_decomps-decomp_level)
    # Rescale pixel intensities to original range
    img_slice = decoder_apply_rescale_attributes(img_slice, scl_intercept, scl_slope)
    decode_time += decode_time_slice
    img += [img_slice]
  # Restitch 3D volume
  img = np.transpose(img, axes=[1,2,0])
  # Save npz (compressed) file 
  if decomp_level == None:
    output_path = os.path.join(
      output_dir, 
      f'npy_scan_{num_decomps+1}',
      strip_extension(path) + '.npz'
    )
  else:
    output_path = os.path.join(
      output_dir, 
      f'npy_scan_{decomp_level+1}',
      strip_extension(path) + '.npz'
    )
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  np.savez_compressed(output_path, img)
  # Return decoding metadata
  return [path, total_slices, decode_time]

def decode3d(
  paths,
  decomp_level,
  input_dir,
  output_dir = None,
  use_multiprocessing = True,
  max_workers = -1
):
  if isinstance(paths, str):
    paths = [paths]
  if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
    raise ValueError('Invalid type! For single-volume decoding, `paths` must be a string and for multi-volume decoding, `paths` must be a list of strings')
  # Default output directory to subdirectory in input directory
  if not output_dir:
    output_dir = input_dir
  encode_df = pd.read_csv(os.path.join(input_dir, 'encode_metadata.csv'))
  if decomp_level == None:
    decomp_level = encode_df['num_decomps'].max()
  # encode_df = encode_df[encode_df['image'].isin(paths)][['image', 'total_slices', 'num_decomps', 'rescale_intensity', 'min_intensity', 'max_intensity']].drop_duplicates('image')
  encode_df = encode_df[encode_df['image'].isin(paths)][['image', 'total_slices', 'num_decomps', 'scl_intercept', 'scl_slope']].drop_duplicates('image')
  data = list(encode_df.itertuples(index=False, name=None))
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
    print(f'Using {max_workers} worker(s) to decode')
    decode_metadata = process_map(partial(__decode3d_worker, decomp_level=decomp_level, input_dir=input_dir, output_dir=output_dir), data, max_workers=max_workers, chunksize=1)
  else:
    decode_metadata = []
    for d in data:
      decode_metadata += [__decode3d_worker(d, decomp_level, input_dir, output_dir)]
  # Save encoding metadata
  pd.DataFrame(
    np.array(decode_metadata, dtype=object),
    columns = ['image', 'total_slices', 'decode_time']
  ).sort_values('image').to_csv(os.path.join(output_dir, f'decode_{decomp_level+1}_metadata.csv'), index=False)