'''
MIST Encoder

Implementation of the MIST encoder. Encodes ingested 2D/3D imaging series (DICOM, NifTi, etc.) for indexing as MIST collection using HTJ2K.

Author(s): Pranav Kulkarni
'''
import os
import numpy as np
import subprocess
import json
import multiprocessing
from functools import partial
import tempfile
from tqdm.contrib.concurrent import process_map

from openjphpy import backend

def encode2D(
  img,
  instance_path,
):
  '''
  Encodes 2D imaging data for MIST collection.
  '''
  raise NotImplementedError('`encode2D` has not been implemented yet.')

def encode3D(
  img,
  instance_paths,
  use_multiprocessing = True,
  max_workers = None,
):
  '''
  Encodes 3D imaging data for MIST collection.
  '''
  raise NotImplementedError('`encode3D` has not been implemented yet.')