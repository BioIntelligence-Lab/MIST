'''
MIST Decoder

Implementation of the MIST decoder. Decodes specified 2D/3D imaging series from MIST collections at requested resolution using HTJ2K.

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

def decode2D(
  instance_path,
  pixel_intensity = None,
  decomposition_level = None,
  resolution_level = None,
  resilient = False,
):
  '''
  Decodes 2D imaging data from MIST collection.
  '''
  raise NotImplementedError('`decode2D` has not been implemented yet.')

def decode3D(
  instance_paths,
  pixel_intensity = None,       # Range [min,max]
  decomposition_level = None,   # Two parameters of skip_res, but inverted
  resolution_level = None,      # ^
  resilient = False,            # From openjph decoder
  use_multiprocessing = True,
  max_workers = None,
):
  '''
  Decodes 3D imaging data from MIST collection.
  '''
  raise NotImplementedError('`decode3D` has not been implemented yet.')