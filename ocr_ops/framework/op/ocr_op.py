from typing import List, Tuple, Optional

import numpy as np
import pytesseract
from pytesseract import Output
from shapely.geometry import box, Polygon

from ocr_ops.framework.op.abstract_ocr_op import (
    PyTesseractOp,
    TextOCROp,
    TextBoxOCROp,
    OCRResult,
    EasyOCROp,
)
from ocr_ops.framework.struct.ocr_result import TextBox


class PyTesseractTextOCROp(PyTesseractOp, TextOCROp):
    """
    PyTesseract Op that only returns OCR-ed txt.
    """

    def __init__(
        self,
        supported_languages: Tuple[str] = ("eng",),
        autosave_img_path: Optional[str] = None,
    ):
        """
        param supported_languages: The languages to support in OCR
        param autosave_img_path: If specified, the place where OCR output images will be auto-saved.
        """
        super().__init__(
            supported_languages=supported_languages, autosave_img_path=autosave_img_path
        )

    def run_ocr(self, img: np.array) -> OCRResult:
        """
        Runs OCR and returns OCRResult.

        param img: Input image

        return:
            OCRResult
        """
        ocr_output: List[str] = [self._image_to_string(img=img)]
        output: OCRResult = OCRResult.from_text_list(texts=ocr_output, input_img=img)
        return output


class PyTesseractTextBoxOCROp(PyTesseractOp, TextBoxOCROp):
    """
    PyTesseract Op that returns full TextBox information.
    """

    def __init__(
        self,
        supported_languages: Tuple[str] = ("eng",),
        autosave_img_path: Optional[str] = None,
    ):
        """
        param supported_languages: The languages to support in OCR
        """
        super().__init__(
            supported_languages=supported_languages, autosave_img_path=autosave_img_path
        )

    def run_ocr(self, img: np.array) -> OCRResult:
        """
        Runs OCR and returns OCRResult.

        param img: Input image

        return:
            OCRResult
        """
        ocr_outputs = pytesseract.image_to_data(img, output_type=Output.DICT)
        text_boxes: List[TextBox] = list()
        for index in range(len(ocr_outputs["text"])):
            text = ocr_outputs["text"][index]
            left = int(ocr_outputs["left"][index])
            top = int(ocr_outputs["top"][index])
            width = int(ocr_outputs["width"][index])
            height = int(ocr_outputs["height"][index])
            conf = float(ocr_outputs["conf"][index])
            bounding_box = box(
                minx=left, miny=top, maxx=(left + width), maxy=(top + height)
            )
            text_box = TextBox(text=text, bounding_box=bounding_box, conf=conf)
            text_boxes.append(text_box)
        output = OCRResult(text_boxes=text_boxes, input_img=img, use_bounding_box=True)
        return output


class EasyOCRTextOp(EasyOCROp, TextOCROp):
    """
    EasyOCR Op that only returns OCR-ed text.
    """

    def __init__(
        self,
        supported_languages: Tuple[str] = ("en",),
        autosave_img_path: Optional[str] = None,
    ):
        """
        param supported_languages: The languages to support in OCR
        param autosave_img_path: If specified, the place where OCR output images will be auto-saved.
        """
        super().__init__(
            supported_languages=supported_languages, autosave_img_path=autosave_img_path
        )

    def run_ocr(self, img: np.array) -> OCRResult:
        """
        Runs OCR and returns OCRResult.

        param img: Input image

        return:
            OCRResult
        """
        ocr_outputs = self._run_easy_ocr(img=img, detail=0)
        output: OCRResult = OCRResult.from_text_list(texts=ocr_outputs, input_img=img)
        return output


class EasyOCRTextBoxOp(EasyOCROp, TextBoxOCROp):
    """
    EasyOCR Op that returns full TextBox information.
    """

    def __init__(
        self,
        supported_languages: Tuple[str] = ("en",),
        autosave_img_path: Optional[str] = None,
    ):
        """
        param supported_languages: The languages to support in OCR
        param autosave_img_path: If specified, the place where OCR output images will be auto-saved.
        """
        super().__init__(
            supported_languages=supported_languages, autosave_img_path=autosave_img_path
        )

    def run_ocr(self, img: np.array) -> OCRResult:
        """
        Runs OCR and returns OCRResult.

        param img: Input image

        return:
            OCRResult
        """
        ocr_outputs = self._run_easy_ocr(img=img, detail=1)
        text_boxes: List[TextBox] = list()
        for ocr_output in ocr_outputs:
            text = ocr_output[1]
            minx = ocr_output[0][0][0]
            miny = ocr_output[0][0][1]
            maxx = ocr_output[0][2][0]
            maxy = ocr_output[0][2][1]
            conf = ocr_output[2]
            bounding_box: Polygon = box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            text_box = TextBox(text=text, bounding_box=bounding_box, conf=conf)
            text_boxes.append(text_box)
        output = OCRResult(text_boxes=text_boxes, input_img=img, use_bounding_box=True)
        return output
