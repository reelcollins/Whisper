import os
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import List

import numpy as np
import torch
from transformers import GPT2TokenizerFast

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}


# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


@dataclass(frozen=True)
class Tokenizer:
    """A thin wrapper around `GPT2TokenizerFast` providing quick access to special tokens"""

    tokenizer: "GPT2TokenizerFast"
    language: Optional[str]
    sot_sequence: Tuple[int]

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: Union[int, List[int], np.ndarray, torch.Tensor], **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, tokens) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        outputs = [[]]
        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        return "".join(outputs)

    @cached_property
    def eot(self) -> int:
        return self.tokenizer.eos_token_id
    
    @cached_property
    def sot(self) -> int:
        return self._get_single_token_id("<|startoftranscript|>")
    
    @cached_property
    def sot_lm(self) -> int:
        return self._get_single_token_id("<|startoflm|>")
    
    @cached_property
    def sot_prev(self) -> int:
        return self._get_single_token_id("<|startofprev|>")
    
    @cached_property
    def no_speech(self) -> int:
        return self._get_single_token_id("<|nospeech|>")
    
    @cached_property
    def timestamp_begin(self) -> int:
        return self.tokenizer.all_special_ids[-1] + 1
    
    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the language field"""
        if self.language is None:
            raise ValueError(f"This tokenizer does not have language token configured")
        
        additional_tokens = dict(
            zip(
                self.tokenizer.additional_special_tokens,
                self.tokenizer.additional_special_tokens_ids,
            )
        )
        candidate = f"<|{self.language}|>"
        if candidate in additional_tokens:
            return additional_tokens[candidate]
        
        raise KeyError(f"language {self.language} not found in tokenizer.")
    
    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in zip(
            self.tokenizer.additional_special_tokens,
            self.tokenizer.additional_special_tokens_ids,
        ):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)
    
    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([1]).strip("<|>") for l  in self.all_language_tokens)
    
    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])
    
    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.
         
        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuation like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they shate the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0*2640 <= ord(c) <= 0*267F for c in miscelleneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.tokenizer.encode(" -")[0], self.tokenizer.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.tokenizer.encode(symbol), self.tokenizer.encoder(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(token[0])

        return tuple(sorted(result))
    
    def _get_single_token_id(self, text) -> int:
        tokens = self.tokenizer.encode(text)
        assert len(tokens) == 1, f"{text} is not encoded as a single token"
        return tokens[0]