from enum import Enum
from typing import Optional, Dict, Any, List, Union

from algo_ops.dependency.sys_util import get_image_files
from algo_ops.ops.op import Op
from algo_ops.paraloop import paraloop
from algo_ops.pipeline.cv_pipeline import CVPipeline
from algo_ops.pipeline.pipeline import Pipeline

from ocr_ops.framework.op.abstract_ocr_op import AbstractOCROp
from ocr_ops.framework.op.ocr_op import (
    PyTesseractTextOCROp,
    PyTesseractTextBoxOCROp,
    EasyOCRTextBoxOp,
    EasyOCRTextOp,
)


class OCRMethod(Enum):
    """
    OCR Method to use for OCR-ing text from an image.
    """

    EASYOCR = 0
    PYTESSERACT = 1


class OutputType(Enum):
    """
    The type of output to obtain from OCR in an OCRResult.
    """

    # just raw text
    TEXT = 0
    # textbox information w/ bounding boxes
    TEXTBOX = 1


class OCRPipeline(Pipeline):
    """
    OCR Pipeline supports running various OCR methods on an image to generate text. It supports
    using a CVOps image pre-processing pipeline to prepare an image for OCR. It also supports a
    TextOps post-processing pipeline to clean noisy OCR-ed text results to return a final robust
    call set of OCR-ed text from an image.
    """

    @staticmethod
    def __setup_ocr_op(ocr_method: OCRMethod, output_type: OutputType) -> AbstractOCROp:
        if ocr_method == OCRMethod.EASYOCR and output_type == OutputType.TEXT:
            return EasyOCRTextOp()
        elif ocr_method == OCRMethod.EASYOCR and output_type == OutputType.TEXTBOX:
            return EasyOCRTextBoxOp()
        elif ocr_method == OCRMethod.PYTESSERACT and output_type == OutputType.TEXT:
            return PyTesseractTextOCROp()
        elif ocr_method == OCRMethod.PYTESSERACT and output_type == OutputType.TEXTBOX:
            return PyTesseractTextBoxOCROp()
        else:
            raise ValueError(
                "Unknown OCR Mode: " + str([str(ocr_method), str(output_type)])
            )

    def __init__(
            self,
            img_pipeline: Optional[CVPipeline],
            ocr_method: OCRMethod,
            output_type: OutputType,
            text_pipeline: Optional[Pipeline],
    ):
        """
        param img_pipeline: An optional CVOps pre-processing pipeline to run on image before OCR
        param ocr_method: The ocr method to use
        param output_type: The type (verbosity) of information output from OCR
        param text_pipeline: An optional TextOps pipeline to post-process OCR text
        """
        self.img_pipeline = img_pipeline
        self.ocr_op = self.__setup_ocr_op(
            ocr_method=ocr_method, output_type=output_type
        )
        self.text_pipeline = text_pipeline

        # prepare ops list
        ops: List[Op] = list()
        # image preprocessing steps
        if self.img_pipeline is not None:
            ops.append(self.img_pipeline)
        # actual OCR on image
        ops.append(self.ocr_op)
        # text cleaning post-processing
        if self.text_pipeline is not None:
            ops.append(self.text_pipeline)
        super().__init__(ops=ops)

    def set_img_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of CVOPs processing pipeline.

        param func_name: The function name in CVOPs pipeline
        param params: Dict mapping function param -> value
        """
        if self.img_pipeline is None:
            raise ValueError("Cannot set parameters when img_pipeline=None.")
        self.img_pipeline.set_pipeline_params(func_name=func_name, params=params)

    def set_text_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of CVOPs processing pipeline.

        param func_name: The function name in CVOPs pipeline
        param params: Dict mapping function param -> value
        """
        if self.text_pipeline is None:
            raise ValueError("Cannot set parameters when text_pipeline=None.")
        self.text_pipeline.set_pipeline_params(func_name=func_name, params=params)

    def run_on_images(
            self, images_dir: str, mechanism: str = "sequential"
    ) -> Union[List[str], List[List[str]]]:
        """
        API to run OCR on a directory of images.

        param files_path: Path to directory of card image files

        return:
            output: List of OCR results
        """
        files = get_image_files(images_dir=images_dir)
        results = paraloop.loop(func=self.exec, params=files, mechanism=mechanism)
        return results
