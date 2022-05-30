import functools
import re
from typing import List, Set, Union, Callable

from algo_ops.ops.text import TextOp
from algo_ops.pipeline.pipeline import Pipeline
from spellchecker import SpellChecker

from ocr_ops.framework.struct.ocr_result import OCRResult


def _extract_confident_text_from_ocr_result(
        ocr_result: Union[OCRResult, str, List[str]]
) -> List[str]:
    if isinstance(ocr_result, str):
        return [ocr_result]
    if isinstance(ocr_result, list):
        return ocr_result

    # remove negative confidence score text boxes
    confident_text = [
        textbox.text.strip()
        for textbox in ocr_result
        if (textbox.conf is None or textbox.conf >= 0.0)
    ]
    return confident_text


def _tokenize_text(text: str) -> List[str]:
    # Tokenize text into words and strip white space.
    words = [w.strip() for w in text.lower().strip().split(" ") if len(w.strip()) > 0]
    return words


def _resplit_new_lines(words: List[str]) -> List[str]:
    # Re-split any existing new lines into words.
    new_words: List[str] = list()
    for word in words:
        if "\n" in word:
            new_words.extend(
                [w.strip() for w in word.split("\n") if len(w.strip()) > 0]
            )
        else:
            new_words.append(word)
    return new_words


def _retokenize_text(text: List[str]) -> List[str]:
    # retokenizes text into words
    all_words: List[str] = list()
    for phrase in text:
        words = _tokenize_text(text=phrase)
        words = _resplit_new_lines(words=words)
        all_words.extend(words)
    return all_words


def _strip(words: List[str]) -> List[str]:
    # remove white space / punctuation
    # reduce to only alphanumeric characters
    stripped = [re.sub("[^a-z0-9]+", "", word) for word in words]
    return stripped


def _correct_spelling(words: List[str]) -> List[str]:
    # attempt to spell check and correct words
    # remove misspellings that cannot be corrected
    spell = SpellChecker()
    incorrect = spell.unknown(words)
    new_words: List[str] = list()
    for i, word in enumerate(words):
        if word in incorrect:
            correction = spell.correction(word)
            if correction != word:
                new_words.append(correction)
        else:
            new_words.append(word)
    return new_words


def _check_vocab(words: List[str], vocab_words: Set[str]) -> List[str]:
    # removes words that do not exist in vocab
    return [word for word in words if word in vocab_words]


class OCRResultUpdater:
    @classmethod
    def _updater(cls, function):
        @functools.wraps(function)
        def wrap(ocr_result: Union[OCRResult, List[str], str], *args, **kwargs):
            if isinstance(ocr_result, OCRResult):
                for text_box in ocr_result.text_boxes:
                    text_box.words = function(text_box.words, *args, **kwargs)
                ocr_result.update_words()
            elif isinstance(ocr_result, list):
                ocr_result = function(ocr_result, *args, **kwargs)
            elif isinstance(ocr_result, str):
                ocr_result = function([ocr_result], *args, **kwargs)
            else:
                raise ValueError("Unknown type of OCRResult: " + str(type(ocr_result)))
            return ocr_result

        return wrap

    @classmethod
    def prepare_updater(cls, base_func: Callable) -> Callable:
        return cls._updater(base_func)


def basic_text_cleaning_pipeline() -> Pipeline:
    pipeline = Pipeline.init_from_funcs(
        [
            OCRResultUpdater.prepare_updater(base_func=base_func)
            for base_func in [_retokenize_text, _strip, _check_vocab]
        ],
        op_class=TextOp,
    )
    return pipeline
