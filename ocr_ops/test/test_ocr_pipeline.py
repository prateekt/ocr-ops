import os
import unittest

import numpy as np
from algo_ops.dependency.iter_params import iter_params
from shapely.geometry import Polygon

from ocr_ops.framework.pipeline.ocr_pipeline import OCRPipeline, OCRMethod, OutputType
from ocr_ops.framework.struct.ocr_result import OCRResult, TextBox
from ocr_ops.instances.cv import black_text_cv_pipeline, white_text_cv_pipeline
from ocr_ops.instances.ocr import (
    basic_ocr_pipeline,
    basic_ocr_with_text_cleaning_pipeline,
    black_text_ocr_pipeline,
    white_text_ocr_pipeline,
)
from ocr_ops.instances.text import basic_text_cleaning_pipeline


class TestOCRPipeline(unittest.TestCase):
    def setUp(self) -> None:
        # paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.joy_of_data_img = os.path.join(dir_path, "data", "joy_of_data.png")

    def test_ocr_pytesseract_text_pipeline(self) -> None:
        """
        Test PyTesseract OCR text-only pipeline.
        """
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.PYTESSERACT,
            output_type=OutputType.TEXT,
            text_pipeline=None,
        )
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertFalse(output.use_bounding_box)
        self.assertTrue(np.array_equal(output.input_img, output.output_img))
        self.assertEqual(len(output), 1)
        self.assertTrue(isinstance(output[0], TextBox))
        self.assertEqual(output[0].text, "joy of data\n")
        self.assertEqual(output[0].bounding_box, None)
        self.assertEqual(output[0].conf, None)

    def test_ocr_pytesseract_textbox_pipeline(self) -> None:
        """
        Test PyTesseract OCR TextBox pipeline.
        """
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.PYTESSERACT,
            output_type=OutputType.TEXTBOX,
            text_pipeline=None,
        )
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertTrue(output.use_bounding_box)
        self.assertFalse(np.array_equal(output.input_img, output.output_img))
        self.assertEqual(len(output), 7)
        output = [a for a in output if a.conf != -1.0]
        self.assertEqual(len(output), 3)
        for i, word in enumerate(["joy", "of", "data"]):
            self.assertTrue(isinstance(output[i], TextBox))
            self.assertEqual(output[i].text, word)
            self.assertTrue(isinstance(output[i].bounding_box, Polygon))
            self.assertTrue(isinstance(output[i].conf, float))

    def test_easyocr_text_pipeline(self) -> None:
        """
        Test EasyOCR text-only pipeline.
        """
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.EASYOCR,
            output_type=OutputType.TEXT,
            text_pipeline=None,
        )
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertFalse(output.use_bounding_box)
        self.assertTrue(np.array_equal(output.input_img, output.output_img))
        self.assertEqual(len(output), 3)
        for i, word in enumerate(["joy", "of", "data"]):
            self.assertTrue(isinstance(output[i], TextBox))
            self.assertEqual(output[i].text, word)
            self.assertEqual(output[i].bounding_box, None)
            self.assertEqual(output[i].conf, None)

    def test_easyocr_textbox_pipeline(self) -> None:
        """
        Test EasyOCR TextBox pipeline.
        """
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.EASYOCR,
            output_type=OutputType.TEXTBOX,
            text_pipeline=None,
        )
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertTrue(output.use_bounding_box)
        self.assertFalse(np.array_equal(output.input_img, output.output_img))
        self.assertEqual(len(output), 3)
        for i, word in enumerate(["joy", "of", "data"]):
            self.assertTrue(isinstance(output[i], TextBox))
            self.assertEqual(output[i].text, word)
            self.assertTrue(isinstance(output[i].bounding_box, Polygon))
            self.assertTrue(isinstance(output[i].conf, float))

    @iter_params(
        ocr_method=(OCRMethod.EASYOCR, OCRMethod.PYTESSERACT),
        output_type=(OutputType.TEXT, OutputType.TEXTBOX),
    )
    def test_ocr_pipeline_with_basic_text_cleaning(
        self, ocr_method: OCRMethod, output_type: OutputType
    ) -> None:
        """
        Test OCR pipeline with basic text cleaning.
        """
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=ocr_method,
            output_type=output_type,
            text_pipeline=basic_text_cleaning_pipeline(),
        )
        ocr_pipeline.set_text_pipeline_params("_check_vocab", {"vocab_words": {"joy"}})
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertEqual(output.words, ["joy"])

    def test_cvpipeline_instances(self) -> None:
        """
        Test CVPipeline instances.
        """

        # black text pipeline test
        cv_pipeline = black_text_cv_pipeline()
        output = cv_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, np.ndarray))

        # white text pipeline test
        cv_pipeline = white_text_cv_pipeline()
        output = cv_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, np.ndarray))

    def test_textpipeline_instances(self) -> None:
        """
        Test text cleaning pipeline instances.
        """

        # basic cleaning pipeline
        text_pipeline = basic_text_cleaning_pipeline()
        text_pipeline.set_pipeline_params(
            "_check_vocab", {"vocab_words": {"joy", "of", "data"}}
        )
        output = text_pipeline.exec("joy of ***%$## data opfu")
        self.assertEqual(output, ["joy", "of", "data"])
        output = text_pipeline.exec(["joy of   \n", "***%$## \n" "data\n opfu\n\n\n\n"])
        self.assertEqual(output, ["joy", "of", "data"])
        text_pipeline.set_pipeline_params("_check_vocab", {"vocab_words": {"data"}})
        output = text_pipeline.exec(["joy of   \n", "***%$## \n" "data\n opfu\n\n\n\n"])
        self.assertEqual(output, ["data"])

    def test_ocr_pipeline_instances(self) -> None:
        """
        Test all OCR pipeline instances.
        """

        # basic pipeline
        p1 = basic_ocr_pipeline()
        output = p1.exec(inp=self.joy_of_data_img)
        self.assertEqual(output[0].text.strip(), "joy of data")

        # basic pipeline with text cleaning
        p2 = basic_ocr_with_text_cleaning_pipeline(vocab_words={"joy", "of"})
        output = p2.exec(inp=self.joy_of_data_img)
        self.assertEqual(output.words, ["joy", "of"])

        # black text ocr
        p3 = black_text_ocr_pipeline()
        p3.set_text_pipeline_params(
            "_check_vocab", {"vocab_words": {"joy", "data", "of"}}
        )
        output = p3.exec(inp=self.joy_of_data_img)
        self.assertEqual(output.words, [])

        # white text ocr
        p4 = white_text_ocr_pipeline()
        p4.set_text_pipeline_params("_check_vocab", {"vocab_words": {"data"}})
        output = p4.exec(inp=self.joy_of_data_img)
        self.assertEqual(output.words, ["data"])
