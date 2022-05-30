from typing import Set

import cv2
import numpy as np
from algo_ops.pipeline.cv_pipeline import CVPipeline

from ocr_ops.framework.pipeline.ocr_pipeline import OCRPipeline, OCRMethod, OutputType
from ocr_ops.instances.text import basic_text_cleaning_pipeline


def basic_ocr_pipeline() -> OCRPipeline:
    """
    Initializes basic PyTesseract OCR pipeline.
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=None,
        ocr_method=OCRMethod.PYTESSERACT,
        output_type=OutputType.TEXT,
        text_pipeline=None,
    )
    return ocr_pipeline


def basic_ocr_with_text_cleaning_pipeline(
    vocab_words: Set[str],
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes basic PyTesseract pipeline with basic text cleaning pipeline.
    """
    img_pipeline = CVPipeline.init_from_funcs(funcs=[_gray_scale])
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        output_type=OutputType.TEXTBOX,
        text_pipeline=basic_text_cleaning_pipeline(),
    )
    ocr_pipeline.set_text_pipeline_params(
        func_name="_check_vocab", params={"vocab_words": vocab_words}
    )
    return ocr_pipeline


def _invert_black_channel(img: np.array) -> np.array:
    # extract black channel in CMYK color space
    # (after this transformation, it appears white)
    img_float = img.astype(np.float) / 255.0
    k_channel = 1 - np.max(img_float, axis=2)
    k_channel = (255 * k_channel).astype(np.uint8)
    return k_channel


def _gray_scale(img: np.array) -> np.array:
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def _remove_background(img: np.array, lower_lim: int = 190) -> np.array:
    # remove background that is not white
    _, bin_img = cv2.threshold(img, lower_lim, 255, cv2.THRESH_BINARY)
    return bin_img


def _invert_back(img: np.array) -> np.array:
    # Invert back to black text / white background
    inv_img = cv2.bitwise_not(img)
    return inv_img


def black_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR black text.
    """

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_invert_black_channel, _remove_background, _invert_back]
    )
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        output_type=OutputType.TEXTBOX,
        text_pipeline=basic_text_cleaning_pipeline(),
    )
    return ocr_pipeline


def white_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR white text.
    """

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_gray_scale, _remove_background, _invert_back]
    )
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        output_type=OutputType.TEXTBOX,
        text_pipeline=basic_text_cleaning_pipeline(),
    )
    return ocr_pipeline
