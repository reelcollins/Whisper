import os
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

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
  open an audio file and read as mono waveform, resampling as necessary

  Parameters
  ----------
  file: str
    The audio file to open
  
  str: int
    The sample rate to resample the audion if necessary


  Returns
  -------
  A NumPy array containing the audio waveform, in float32 dtype.
  
  """
  try:
    # This launches a subprocess to decide audio while down-mixing and resampling as necessary.
    # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
    out, _ = (
      ffmpeg.input(file, threads=0)
      .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
      .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
  except ffmpeg.Error as e:
    raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
  
  return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0



def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
  """
  Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
  """
  if torch.is_tensor(array):
    if array.shape[axis] > length:
      array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

    if array.shape[axis] < length:
      pad_widths = [(0, 0)] * array.ndim
      pad_widths[axis] = (0, length - array.shape[axis])
      array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])