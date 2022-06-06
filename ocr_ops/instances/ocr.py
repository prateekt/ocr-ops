from typing import Set, Optional

from ocr_ops.framework.pipeline.ocr_pipeline import OCRPipeline, OCRMethod, OutputType
from ocr_ops.instances.cv import (
    black_text_cv_pipeline,
    white_text_cv_pipeline,
    basic_cv_pipeline,
)
from ocr_ops.instances.text import basic_text_cleaning_pipeline


def basic_ocr_pipeline(autosave_output_img_path: Optional[str] = None) -> OCRPipeline:
    """
    Initializes basic PyTesseract OCR pipeline.

    param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.

    return:
        OCRPipeline
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=None,
        ocr_method=OCRMethod.PYTESSERACT,
        output_type=OutputType.TEXT,
        text_pipeline=None,
        autosave_output_img_path=autosave_output_img_path,
    )
    return ocr_pipeline


def basic_ocr_with_text_cleaning_pipeline(
    vocab_words: Set[str],
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
    autosave_output_img_path: Optional[str] = None,
) -> OCRPipeline:
    """
    Initializes basic PyTesseract OCR pipeline.

    param vocab_words: The set of vocab words to accept
    param ocr_method: The OCR Method to use
    param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.

    return:
        OCRPipeline
    """
    img_pipeline = basic_cv_pipeline()
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        output_type=OutputType.TEXTBOX,
        text_pipeline=basic_text_cleaning_pipeline(),
        autosave_output_img_path=autosave_output_img_path,
    )
    ocr_pipeline.set_text_pipeline_params(
        func_name="_check_vocab", params={"vocab_words": vocab_words}
    )
    return ocr_pipeline


def black_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
    autosave_output_img_path: Optional[str] = None,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR black text.

    param ocr_method: The OCR Method to use
    param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.

    return:
        OCRPipeline
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=black_text_cv_pipeline(),
        ocr_method=ocr_method,
        output_type=OutputType.TEXTBOX,
        text_pipeline=basic_text_cleaning_pipeline(),
        autosave_output_img_path=autosave_output_img_path,
    )
    return ocr_pipeline


def white_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
    autosave_output_img_path: Optional[str] = None,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR white text.

    param ocr_method: The OCR Method to use
    param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.

    return:
        OCRPipeline
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=white_text_cv_pipeline(),
        ocr_method=ocr_method,
        output_type=OutputType.TEXTBOX,
        text_pipeline=basic_text_cleaning_pipeline(),
        autosave_output_img_path=autosave_output_img_path,
    )
    return ocr_pipeline
