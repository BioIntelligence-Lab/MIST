import os
import multiprocessing
import numpy as np
from functools import partial
import nibabel as nib
from tqdm.contrib.concurrent import process_map
from . import utils

def __npy2nifti_worker(
  path,
  nifti_dir,
  input_dir,
  output_dir
):
  # Load header and affine from source NII
  ds = nib.load(os.path.join(nifti_dir, path))
  # Load npz volume decoded from HTJ2K and covert to NII dtype
  img = np.load(os.path.join(input_dir, utils.strip_extension(path) + '.npz'))['arr_0'].astype(f"{ds.header.get_data_dtype()}")
  # Num of data dims
  n = ds.header['dim'][0]
  # Source data shape
  src_dim = ds.header['dim'][1:n+1]
  # Source data voxel size
  src_vox_size = ds.header['pixdim'][1:n+1]
  # Target data shape
  target_dim = np.array(img.shape)
  # Target data voxel size as factor of source voxel size
  target_vox_size = src_vox_size*src_dim/target_dim
  # Rescale affine to target data shape and voxel size
  target_affine = nib.affines.rescale_affine(
    ds.affine, 
    src_dim, 
    target_vox_size, 
    target_dim
  )
  os.makedirs(os.path.join(output_dir, os.path.dirname(path)), exist_ok=True)
  if isinstance(ds, nib.Nifti1Image):
    target_ds = nib.Nifti1Image(img, target_affine, ds.header)
    nib.save(
      target_ds,
      os.path.join(output_dir, path)
    )
  elif isinstance(ds, nib.Nifti2Image):
    target_ds = nib.Nifti2Image(img, target_affine, ds.header)
    nib.save(
      target_ds,
      os.path.join(output_dir, path)
    )
  else:
    raise NotImplementedError('Unknown NifTI format')

def npy2nifti(
  paths,
  nifti_dir,
  input_dir,
  output_dir = None,
  use_multiprocessing = True,
  max_workers = -1
):
  if isinstance(paths, str):
    paths = [paths]
  if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
    raise ValueError('Invalid type! For single-volume conversion, `paths` must be a string and for multi-volume conversion, `paths` must be a list of strings')
  # Default output directory to subdirectory in input directory
  if not output_dir:
    output_dir = input_dir
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
    print(f'Using {max_workers} worker(s) to convert')
    process_map(partial(__npy2nifti_worker, nifti_dir=nifti_dir, input_dir=input_dir, output_dir=output_dir), paths, max_workers=max_workers, chunksize=1)
  else:
    for path in paths:
      __npy2nifti_worker(path, nifti_dir, input_dir, output_dir)
