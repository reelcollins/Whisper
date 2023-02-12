import hashlib
import io
import os
import urlib
import warnings
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from .audio import load_audio, log_mel_spectogram, pad_or_trim
from .decoding import DecodingOptions, DecodingResults, decode, detect_language
from .model import Whisper, ModelDimensions
from .transcribe import transcribe
from .version import __version__


_MODELS = {

}
