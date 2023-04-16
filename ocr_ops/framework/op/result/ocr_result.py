from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
from algo_ops.ops.cv import ImageResult
from shapely.geometry import Polygon


class TextBox:
    """
    Represents a block of OCR-ed text containing one or multiple words in an image.
    """

    def __init__(
        self,
        text: str,
        bounding_box: Optional[Polygon] = None,
        conf: Optional[float] = None,
    ):
        """
        param text: OCR-ed text
        param bounding_box: The bounding box of the test
        param conf: OCR Confidence Score
        """
        self.text: str = text
        self.bounding_box: Optional[Polygon] = bounding_box
        self.conf: Optional[float] = conf
        self.words: List[str] = [text]

    def __str__(self):
        if self.bounding_box is not None and self.conf is not None:
            return str([self.text, self.bounding_box.bounds, self.conf])
        else:
            return str([self.text])

    def overlap_area(self, other: "TextBox"):
        """
        Finds the overlap area between one text box's bounding box with another.

        param other: Other TextBox

        return:
           Area (in pixels) of overlap amount
        """
        if self.bounding_box is None:
            raise ValueError(
                "No bounding box is specified in this DetectedText object: " + str(self)
            )
        if other.bounding_box is None:
            raise ValueError(
                "No bounding box found in other DetectedText object: " + str(other)
            )
        area = self.bounding_box.intersection(other.bounding_box).area
        return area

    def percent_overlap(self, other: "TextBox"):
        """
        % overlap of one textbox with another.

        param other: Other TextBox

        return:
            % overlap
        """
        if self.bounding_box is None:
            raise ValueError(
                "No bounding box is specified in this DetectedText object: " + str(self)
            )
        if other.bounding_box is None:
            raise ValueError(
                "No bounding box found in other DetectedText object: " + str(other)
            )
        return self.overlap_area(other=other) / self.bounding_box.area


class OCRImageResult:
    """
    OCRImageResult holds the result of an OCR operation on a single image.
    """

    def __init__(
        self, text_boxes: List[TextBox], input_img: ImageResult, use_bounding_box: bool
    ):
        """
        param text_boxes: Detected text boxes in image
        param input_img: Pointer to raw input image
        param use_bounding_box: Whether bounding box annotations are used
        """
        self.text_boxes: List[TextBox] = text_boxes
        self.input_img: Optional[ImageResult] = input_img
        self.use_bounding_box = use_bounding_box
        self.output_img: Optional[np.array] = self._prepare_output_image()
        self.words: List[str] = list()
        self.update_words()

    def __str__(self):
        return str([str(text_box) for text_box in self.text_boxes])

    def _prepare_output_image(self) -> np.array:
        """
        Overlay text bounding boxes on input image for visualization.

        return:
            output_image: Output image ready for visualization
        """
        output_img = self.input_img.img.copy()
        if self.use_bounding_box:
            for text_box in self.text_boxes:
                (x0, y0, xf, yf) = [int(a) for a in text_box.bounding_box.bounds]
                cv2.rectangle(output_img, (x0, y0), (xf, yf), (0, 255, 0), 2)
        return output_img

    def update_words(self) -> None:
        """
        Updates words with the latest information from text boxes.
        """
        words: List[str] = list()
        for text_box in self.text_boxes:
            words += text_box.words
        self.words = words

    @staticmethod
    def from_text_list(texts: List[str], input_img: np.array) -> "OCRImageResult":
        """
        Creates an OCRImageResult from a list of detected text strings (no bounding box annotations)

        param texts: List of detected text strings
        param input_img: The input image

        return:
            ocr_result: OCRImageResult object
        """
        text_boxes: List[TextBox] = [TextBox(text=text) for text in texts]
        ocr_result = OCRImageResult(
            text_boxes=text_boxes, input_img=input_img, use_bounding_box=False
        )
        return ocr_result

    def __getitem__(self, i) -> TextBox:
        return self.text_boxes[i]

    def __setitem__(self, key, value) -> None:
        self.text_boxes[key] = value

    def __len__(self) -> int:
        return len(self.text_boxes)

    def to_text_list(self, strip: bool = False) -> List[str]:
        """
        Returns just the text detected in TextBoxes as list of strings.

        return:
            Detected text as list of strings
        """
        return [
            text_box.text.strip() if strip else text_box.text
            for text_box in self.text_boxes
        ]


class OCRPipelineResult:
    """
    Represents the full result of an OCRPipeline task (on one or multiple image frames or on a video).
    """

    def __init__(self, ocr_image_results: List[OCRImageResult], input_path: str):
        """
        param ocr_image_results: List of OCR Image Results, one for each input image frame
        param input_path: Path to original input file or directory
        """
        self.ocr_image_results: List[OCRImageResult] = ocr_image_results
        self.input_path: str = input_path

    def __getitem__(self, i) -> OCRImageResult:
        return self.ocr_image_results[i]

    def __setitem__(self, key, value) -> None:
        self.ocr_image_results[key] = value

    def __len__(self) -> int:
        return len(self.ocr_image_results)

    def to_df(self) -> pd.DataFrame:
        """
        Convert OCR results to pandas dataframe containing one row per detected text box.
        """

        # prepare dataframe of bounding box results
        input_paths = list()
        texts = list()
        bounding_boxes = list()
        confidences = list()
        for ocr_result in self.ocr_image_results:
            for bounding_box in ocr_result.text_boxes:
                if ocr_result.input_img is None:
                    input_paths.append("N/A")
                else:
                    input_paths.append(ocr_result.input_img.file_path)
                texts.append(bounding_box.text)
                bounding_boxes.append(bounding_box.bounding_box)
                confidences.append(bounding_box.conf)
        df = pd.DataFrame(
            {
                "input_path": input_paths,
                "text": texts,
                "bounding_box": bounding_boxes,
                "confidence": confidences,
            }
        )
        return df
