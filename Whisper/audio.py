import os
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import tor ch.nn.functional as F

from utils import exact_div


# hard-coded audion hyperparameters
SAMPLE_RATE = 16000 #  Hz
N_FFT = 400
N_MELS = 80 # channel
HOP_LENGTH = 160
CHUNK_LENGTH = 30 # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES =  exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
  """
  
  """
