import os.path
from enum import Enum
from typing import Optional, Dict, Any, List, Union

from algo_ops.dependency.sys_util import get_image_files, is_image_file
from algo_ops.ops.op import Op
from algo_ops.paraloop import paraloop
from algo_ops.pipeline.cv_pipeline import CVPipeline
from algo_ops.pipeline.pipeline import Pipeline

from ocr_ops.framework.op.abstract_ocr_op import AbstractOCROp, EasyOCROp
from ocr_ops.framework.op.drop_op import DropOp
from ocr_ops.framework.op.ocr_op import (
    PyTesseractTextOCROp,
    PyTesseractTextBoxOCROp,
    EasyOCRTextBoxOp,
    EasyOCRTextOp,
)
from ocr_ops.framework.op.result.ffmeg_result import FFMPEGResult
from ocr_ops.framework.op.result.ocr_result import OCRImageResult, OCRPipelineResult


class OCRMethod(Enum):
    """
    OCR Method to use for OCR-ing text from an image.
    """

    EASYOCR = 0
    PYTESSERACT = 1


class OutputType(Enum):
    """
    The type of output to obtain from OCR in an OCRPipelineResult.
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
    def __setup_ocr_op(
        ocr_method: OCRMethod,
        output_type: OutputType,
        autosave_output_img_path: Optional[str],
    ) -> AbstractOCROp:
        """
        Helper function to set up ocr op.

        param ocr_method: The ocr method to use
        param output_type: The type (verbosity) of information output from OCR
        param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.

        return:
            OCR Op
        """
        if ocr_method == OCRMethod.EASYOCR and output_type == OutputType.TEXT:
            return EasyOCRTextOp(autosave_output_img_path=autosave_output_img_path)
        elif ocr_method == OCRMethod.EASYOCR and output_type == OutputType.TEXTBOX:
            return EasyOCRTextBoxOp(autosave_output_img_path=autosave_output_img_path)
        elif ocr_method == OCRMethod.PYTESSERACT and output_type == OutputType.TEXT:
            return PyTesseractTextOCROp(
                autosave_output_img_path=autosave_output_img_path
            )
        elif ocr_method == OCRMethod.PYTESSERACT and output_type == OutputType.TEXTBOX:
            return PyTesseractTextBoxOCROp(
                autosave_output_img_path=autosave_output_img_path
            )
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
        autosave_output_img_path: Optional[str] = None,
        store_intermediate_images: bool = True,
    ):
        """
        param img_pipeline: An optional CVOps pre-processing pipeline to run on image before OCR
        param ocr_method: The ocr method to use
        param output_type: The type (verbosity) of information output from OCR
        param text_pipeline: An optional TextOps pipeline to post-process OCR text
        param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.
        """
        self.img_pipeline: Optional[CVPipeline] = img_pipeline
        self.ocr_op: AbstractOCROp = self.__setup_ocr_op(
            ocr_method=ocr_method,
            output_type=output_type,
            autosave_output_img_path=autosave_output_img_path,
        )
        self.text_pipeline: Optional[Pipeline] = text_pipeline
        self.parallel_mechanism: str = "sequential"
        self.input: Optional[str] = None
        self.output: Optional[OCRPipelineResult] = None

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
        if not store_intermediate_images:
            ops.append(DropOp())
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

    @staticmethod
    def _parse_image_files_list_from_input(input_path: str) -> List[str]:
        """
        Parse and validate input path.

        param input_path: Input path
        return:
            Image files in path
        """
        input_path = input_path
        if not os.path.exists(input_path):
            raise ValueError("Input_path " + str(input_path) + " does not exist.")
        if os.path.isdir(input_path):
            files = get_image_files(images_dir=input_path)
            if len(files) == 0:
                raise ValueError(
                    "Input_path " + str(input_path) + " contains no image files."
                )
        else:
            if not is_image_file(input_path):
                raise ValueError(
                    "Input_path " + str(input_path) + " is not an image file."
                )
            files = [input_path]
        return files

    def exec(self, inp: Union[str, FFMPEGResult]) -> OCRPipelineResult:
        """
        API to run OCR on a single image or a directory of images.

        param inp: Path to single image file or directory containing image file(s)

        return:
            output: List of OCR results
        """
        # prepare input as List[str] of files
        if isinstance(inp, FFMPEGResult):
            images_path = inp.output_images_path
            original_input_path = inp.input_video_path
        elif isinstance(inp, str):
            images_path = inp
            original_input_path = inp
        else:
            raise ValueError("Unknown input type: " + str(type(inp)))
        files = self._parse_image_files_list_from_input(input_path=images_path)

        # run OCR on each file
        ocr_img_results: List[OCRImageResult] = paraloop.loop(
            func=super().exec, params=files, mechanism=self.parallel_mechanism
        )
        self.input = images_path

        # prepare output and return
        return OCRPipelineResult(
            ocr_image_results=ocr_img_results, input_path=original_input_path
        )

    def to_pickle(self, out_pkl_path: str) -> None:
        """
        Pickle ocr pipeline to pickle file.

        param out_pkl_path: Path to where pickle file should go
        """

        # temporarily remove un-pickleable elements
        easy_ocr_instance = None
        if isinstance(self.ocr_op, EasyOCROp):
            easy_ocr_instance = self.ocr_op.easy_ocr_reader
            self.ocr_op.easy_ocr_reader = None

        # super call to pickle
        super().to_pickle(out_pkl_path=out_pkl_path)

        # restore state
        if isinstance(self.ocr_op, EasyOCROp):
            self.ocr_op.easy_ocr_reader = easy_ocr_instance
