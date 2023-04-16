import os
import tempfile
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any, Optional, Dict

import cv2
import ezplotly.settings as plot_settings
import numpy as np
from algo_ops.ops.cv import ImageResult, CVOp
from algo_ops.ops.op import Op
from algo_ops.plot.plot import pyplot_image
from easyocr import easyocr
from pytesseract import pytesseract, Output

from ocr_ops.framework.op.result.ocr_result import OCRImageResult


class AbstractOCROp(Op, ABC):
    """
    Turns the use of OCR package into an Op.
    """

    def __init__(
        self,
        supported_languages: Tuple[str],
        autosave_output_img_path: Optional[str] = None,
    ):
        """
        Constructor for Abstract OCROp.

        param supported_languages: The languages to support in OCR
        param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.
        """
        super().__init__(func=self.run_ocr)
        self.supported_languages = supported_languages
        self.autosave_output_img_path = autosave_output_img_path
        self.input: Optional[ImageResult] = None
        self.output: Optional[OCRImageResult] = None

    @abstractmethod
    def run_ocr(self, img: ImageResult) -> OCRImageResult:
        """
        Runs OCR pipeline on an image.

        param img: Input Image

        return:
            ocr_result: OCRResultObject
        """
        pass

    def exec(self, inp: Union[str, np.array, ImageResult]) -> OCRImageResult:
        """
        Execute OCR operation on input image to produce OCRPipelineResult.

        param inp: Either path to image file, numpy image matrix, or ImageResult

        return:
            OCR Result
        """

        # parse input into ImageResult object
        input_img_result = CVOp.parse_input(inp=inp)
        if input_img_result.file_path is not None:
            basename: str = os.path.splitext(
                os.path.basename(input_img_result.file_path)
            )[0]
        else:
            basename: str = str(len(self.execution_times)).zfill(6)

        # run function on input image to produce OCR Result
        output = super().exec(input_img_result)
        assert isinstance(output, OCRImageResult)
        self.input = input_img_result
        self.output = output

        # autosave (if necessary)
        if self.autosave_output_img_path is not None:
            self.save_output(out_path=self.autosave_output_img_path, basename=basename)

        # return output
        return self.output

    def vis_input(self) -> None:
        """
        Plot current input image using pyplot (jupyter compatible)
        """
        if self.input is None:
            raise ValueError("There is no input to be visualized.")
        else:
            self.input.plot(title=self.name)

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Saves current input image to file.

        param out_path: Path to where file should be saved.
        param basename: Basename of file
        """
        if self.input is not None:
            if basename is None:
                basename = self.name
            basename += "_input"
            self.input.save(out_path=out_path, basename=basename)
        else:
            raise ValueError("There is no input to be saved.")


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
                os.makedirs(out_path, exist_ok=True)
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
        if plot_settings.SUPPRESS_PLOTS:
            print("Plot of " + str(self.name) + " output suppressed.")
        else:
            pyplot_image(img=self.output.output_img, title=self.name)

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        """
        Save output to file.

        param out_path: Path to where file should go
        param basename: File basename
        """
        if self.output is not None:
            if self.output.output_img is None:
                raise ValueError("Input image is None.")
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
        autosave_output_img_path: Optional[str] = None,
    ):
        """
        param supported_languages: The languages to support in OCR
        param autosave_output_img_path: If specified, the place where OCR output images will be auto-saved.

        """
        super().__init__(
            supported_languages=supported_languages,
            autosave_output_img_path=autosave_output_img_path,
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

    def to_pickle(self, out_pkl_path: str) -> None:
        # temporarily remove un-pickleable elements
        easy_ocr_instance = self.easy_ocr_reader
        self.easy_ocr_reader = None

        # super call to pickle
        super().to_pickle(out_pkl_path=out_pkl_path)

        # restore state
        self.easy_ocr_reader = easy_ocr_instance
