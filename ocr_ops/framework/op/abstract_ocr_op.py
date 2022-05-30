import os
import tempfile
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any, Optional, Dict

import cv2
import numpy as np
from algo_ops.ops.cv import CVOp
from algo_ops.ops.op import Op
from easyocr import easyocr
from pytesseract import pytesseract, Output

from ocr_ops.framework.struct.ocr_result import OCRResult


class AbstractOCROp(Op, ABC):
    """
    Turns the use of OCR package into an Op.
    """

    def vis_input(self) -> None:
        """
        Visualizes input to OCROp. Raises a ValueError if there is no input.
        """
        if self.input_img is None:
            raise ValueError(
                "There is no input to be visualized since "
                + str(self.name)
                + " has not executed yet."
            )
        CVOp.pyplot_image(img=self.input_img, title=self.name)

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves input to file.

        param out_path: Path to where saved file should go
        param basename: Basename of saved file
        """
        if self.input_img is not None:
            if out_path.endswith(".png"):
                outfile = out_path
            else:
                os.makedirs(out_path, exist_ok=True)
                if basename is not None:
                    outfile = os.path.join(out_path, basename + "_input.png")
                else:
                    outfile = os.path.join(out_path, self.name + "_input.png")
            cv2.imwrite(outfile, self.input_img)
        else:
            raise ValueError(
                "There is not input, and Op "
                + str(self.name)
                + " has not executed yet."
            )

    @abstractmethod
    def run_ocr(self, img: np.array) -> OCRResult:
        """
        Runs OCR pipeline on an image.

        param img: Image matrix in numpy

        return:
            ocr_result: OCRResultObject
        """
        pass

    def exec_ocr(self, inp: Union[str, np.array]) -> OCRResult:
        """
        Executes OCR on an input and returns OCRResult. Flexible wrapper that can take in an image file path or image.

        param inp: Either path to image file or numpy image matrix

        return:
            ocr_result: OCRResult
        """
        if isinstance(inp, str):
            img = cv2.imread(filename=inp)
        elif isinstance(inp, np.ndarray):
            img = inp
        else:
            raise ValueError("Unsupported input: " + str(inp))
        self.input_img = img
        ocr_result = self.run_ocr(img=img)
        return ocr_result

    def __init__(
        self,
        supported_languages: Tuple[str],
    ):
        """
        Constructor for Abstract OCROp.

        param supported_languages: The languages to support in OCR
        """
        self.supported_languages = supported_languages
        self.input_img: Optional[np.array] = None
        super().__init__(func=self.exec_ocr)


class TextOCROp(AbstractOCROp, ABC):
    """
    Simple OCROp that only returns a list of detected text strings in an image.
    """

    def vis(self) -> None:
        """
        Print current output.
        """
        print(self.name + ": " + str(self.output))

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current output to file.

        param out_path: Path to where output file should be saved.
        param basename: Basename of output file
        """
        if self.output is not None:
            if out_path.endswith(".txt"):
                outfile = out_path
            else:
                if basename is not None:
                    outfile = os.path.join(out_path, basename + ".txt")
                else:
                    outfile = os.path.join(out_path, self.name + ".txt")
            with open(outfile, "w") as out_file:
                all_text = [text_box.text for text_box in self.output]
                out_file.write("\n".join(all_text))
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")


class TextBoxOCROp(AbstractOCROp, ABC):
    """
    OCR operation that returns detected text as well as text boxes.
    """

    def vis(self) -> None:
        """
        Visualizes output using pyplot (Jupyter compatible)
        """
        if self.output is None:
            raise ValueError(
                "There is no output to be visualized since "
                + str(self.name)
                + " has not executed yet."
            )
        CVOp.pyplot_image(img=self.output.output_img, title=self.name)

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Save output to file.

        param out_path: Path to where file should go
        param basename: File basename
        """
        if self.output is not None:
            if out_path.endswith(".png"):
                outfile = out_path
            else:
                os.makedirs(out_path, exist_ok=True)
                if basename is not None:
                    outfile = os.path.join(out_path, basename + ".png")
                else:
                    outfile = os.path.join(out_path, self.name + ".png")
            cv2.imwrite(outfile, self.output.output_img)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")


class PyTesseractOp(AbstractOCROp, ABC):
    """
    Run PyTesseract as OCRxOp.
    """

    def __prepare_lang(self) -> str:
        """
        Prepare language string.
        """
        lang = "+".join(self.supported_languages)
        return lang

    def _image_to_string(self, img: np.array) -> str:
        """
        Wrapper for PyTesseract image_to_string.

        param img: Input image

        return:
            ocr_outputs: OCR-ed text as string
        """
        ocr_outputs = pytesseract.image_to_string(image=img, lang=self.__prepare_lang())
        return ocr_outputs

    def _image_to_data(self, img: np.array) -> Dict[str, Any]:
        """
        Wrapper for PyTesseract image_to_data.

        param img: Input image

        return:
            ocr_outputs: Output dictionary from PyTesseract
        """
        ocr_outputs = pytesseract.image_to_data(
            img, output_type=Output.DICT, lang=self.__prepare_lang()
        )
        return ocr_outputs


class EasyOCROp(AbstractOCROp, ABC):
    """
    Run EasyOCR as OCROp.
    """

    def __init__(
        self,
        supported_languages: Tuple[str] = ("en",),
    ):
        """
        param supported_languages: The languages to support in OCR
        """
        super().__init__(
            supported_languages=supported_languages,
        )
        self.easy_ocr_reader: Optional[easyocr.Reader] = easyocr.Reader(
            lang_list=list(self.supported_languages)
        )

    def _run_easy_ocr(self, img: np.array, detail: int) -> Any:
        """
        Runs easyocr method on input image.

        param img: Input image object
        detail: 0 for just text, 1 for verbose output with bounding boxes and confidence scores

        return:
            output: OCR Result
        """
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png") as png:
            cv2.imwrite(png.name, img)
            result = self.easy_ocr_reader.readtext(png.name, detail=detail)
        return result
