import json
import os
import sys
import zlib
from typing import Callable, TextIO

def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))