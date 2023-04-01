import os
import unittest

import ezplotly.settings as plot_settings
from algo_ops.dependency.tester_util import clean_paths
from algo_ops.ops.cv import ImageResult

from ocr_ops.framework.op.ocr_op import (
    EasyOCRTextOp,
    PyTesseractTextOCROp,
    EasyOCRTextBoxOp,
    PyTesseractTextBoxOCROp,
)
from ocr_ops.framework.op.result.ocr_result import OCRImageResult


class TestOCROps(unittest.TestCase):
    @staticmethod
    def _clean_env():
        clean_paths(
            dirs=(
                "txt_ocr_output",
                "easy_ocr_profile",
                "pytesseract_profile",
                "box_ocr_output",
                "easy_ocr_autosave",
                "pytesseract_autosave",
            )
        )

    def setUp(self) -> None:
        # paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.joy_of_data_img = os.path.join(dir_path, "data", "joy_of_data.png")
        self.blank_card_img = os.path.join(dir_path, "data", "blank_card.png")

        # env
        self._clean_env()
        plot_settings.SUPPRESS_PLOTS = True

    def tearDown(self) -> None:
        self._clean_env()

    def test_text_ocr(self) -> None:
        """
        Test TextOCROp on test images.
        """

        # init
        easy_ocr_op = EasyOCRTextOp(autosave_output_img_path="easy_ocr_autosave")
        pytesseract_op = PyTesseractTextOCROp(
            autosave_output_img_path="pytesseract_autosave"
        )

        # test that ops without inputs don't do much
        self.assertEqual(easy_ocr_op.input, None)
        self.assertEqual(pytesseract_op.input, None)
        self.assertEqual(easy_ocr_op.output, None)
        self.assertEqual(pytesseract_op.output, None)
        self.assertEqual(len(easy_ocr_op.execution_times), 0)
        self.assertEqual(len(pytesseract_op.execution_times), 0)
        for method in [
            easy_ocr_op.vis_profile,
            easy_ocr_op.save_input,
            easy_ocr_op.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()
        for method in [
            pytesseract_op.vis_profile,
            pytesseract_op.save_input,
            pytesseract_op.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()

        # test easy ocr on input images
        output = easy_ocr_op.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(easy_ocr_op.input, ImageResult))
        self.assertTrue(isinstance(output, OCRImageResult))
        self.assertEqual(output.to_text_list(), ["joy", "of", "data"])
        output = easy_ocr_op.exec(self.blank_card_img)
        self.assertTrue(isinstance(output, OCRImageResult))
        self.assertEqual(output.to_text_list(), [])

        # test pytesseract on test images
        output = pytesseract_op.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(easy_ocr_op.input, ImageResult))
        self.assertTrue(isinstance(output, OCRImageResult))
        self.assertEqual(output.to_text_list(strip=True), ["joy of data"])
        output = pytesseract_op.exec(self.blank_card_img)
        self.assertTrue(isinstance(output, OCRImageResult))
        for autosave_file in ("blank_card.txt", "joy_of_data.txt"):
            self.assertTrue(
                os.path.exists(os.path.join("pytesseract_autosave", autosave_file))
            )
            self.assertTrue(
                os.path.exists(os.path.join("easy_ocr_autosave", autosave_file))
            )

        # test saving input / output
        easy_ocr_op.save_input(out_path="txt_ocr_output", basename="easy_ocr")
        easy_ocr_op.save_output(out_path="txt_ocr_output", basename="easy_ocr")
        pytesseract_op.save_input(out_path="txt_ocr_output", basename="pytesseract_ocr")
        pytesseract_op.save_output(
            out_path="txt_ocr_output", basename="pytesseract_ocr"
        )
        for file in ("easy_ocr", "pytesseract_ocr"):
            self.assertTrue(
                os.path.exists(os.path.join("txt_ocr_output", file + ".txt"))
            )
            self.assertTrue(
                os.path.exists(os.path.join("txt_ocr_output", file + "_input.png"))
            )

        # test visualizing profile
        easy_ocr_op.vis_profile(profiling_figs_path="easy_ocr_profile")
        pytesseract_op.vis_profile(profiling_figs_path="pytesseract_profile")
        self.assertTrue(os.path.exists(os.path.join("easy_ocr_profile", "run_ocr.png")))
        self.assertTrue(
            os.path.exists(os.path.join("pytesseract_profile", "run_ocr.png"))
        )

    def test_textbox_ocr_op(self) -> None:
        """
        Test TextBox OCR Ops on test images.
        """

        # init
        easy_ocr_op = EasyOCRTextBoxOp(autosave_output_img_path="easy_ocr_autosave")
        pytesseract_op = PyTesseractTextBoxOCROp(
            autosave_output_img_path="pytesseract_autosave"
        )

        # test that ops without inputs don't do much
        self.assertEqual(easy_ocr_op.input, None)
        self.assertEqual(pytesseract_op.input, None)
        self.assertEqual(easy_ocr_op.output, None)
        self.assertEqual(pytesseract_op.output, None)
        self.assertEqual(len(easy_ocr_op.execution_times), 0)
        self.assertEqual(len(pytesseract_op.execution_times), 0)
        for method in (
            easy_ocr_op.vis_input,
            easy_ocr_op.vis,
            easy_ocr_op.vis_profile,
            easy_ocr_op.save_input,
            easy_ocr_op.save_output,
        ):
            with self.assertRaises(ValueError):
                method()
        for method in (
            pytesseract_op.vis_input,
            pytesseract_op.vis,
            pytesseract_op.vis_profile,
            pytesseract_op.save_input,
            pytesseract_op.save_output,
        ):
            with self.assertRaises(ValueError):
                method()

        # test that spatial bounding boxes overlap significantly between the two OCR methods for the same detected
        # text on joy of data image
        output1: OCRImageResult = easy_ocr_op.exec(self.joy_of_data_img)
        output2: OCRImageResult = pytesseract_op.exec(self.joy_of_data_img)
        output2.text_boxes = [
            output
            for output in output2
            if output.conf != -1.0 and len(output.text.strip()) > 0
        ]
        self.assertTrue(isinstance(easy_ocr_op.input, ImageResult))
        self.assertTrue(isinstance(pytesseract_op.input, ImageResult))
        self.assertEqual(len(output1), 3)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(all(a.text == b.text for (a, b) in zip(output1, output2)))
        self.assertTrue(
            all(a.percent_overlap > 0.95 and b.percent_overlap > 0.95)
            for (a, b) in zip(output1, output2)
        )
        for i1, a in enumerate(output1):
            for i2, b in enumerate(output2):
                if i1 == i2:
                    self.assertTrue(a.percent_overlap(b) > 0.5)
                    self.assertEqual(b.percent_overlap(a), 1.0)
                else:
                    self.assertEqual(a.percent_overlap(b), 0.0)
                    self.assertEqual(b.percent_overlap(a), 0.0)

        # test saving input / output
        easy_ocr_op.save_input(out_path="box_ocr_output", basename="easy_ocr")
        easy_ocr_op.save_output(out_path="box_ocr_output", basename="easy_ocr")
        pytesseract_op.save_input(out_path="box_ocr_output", basename="pytesseract_ocr")
        pytesseract_op.save_output(
            out_path="box_ocr_output", basename="pytesseract_ocr"
        )
        for file in (
            "easy_ocr",
            "easy_ocr_input",
            "pytesseract_ocr",
            "pytesseract_ocr_input",
        ):
            self.assertTrue(
                os.path.exists(os.path.join("box_ocr_output", file + ".png"))
            )

        # test that nothing is detected in blank image
        output1: OCRImageResult = easy_ocr_op.exec(self.blank_card_img)
        output2: OCRImageResult = pytesseract_op.exec(self.blank_card_img)
        output2.text_boxes = [
            output
            for output in output2
            if output.conf != -1.0 and len(output.text.strip()) > 0
        ]
        self.assertEqual(output1.text_boxes, [])
        self.assertEqual(output2.text_boxes, [])

        # test visualizing profile
        easy_ocr_op.vis_profile(profiling_figs_path="easy_ocr_profile")
        pytesseract_op.vis_profile(profiling_figs_path="pytesseract_profile")
        self.assertTrue(os.path.exists(os.path.join("easy_ocr_profile", "run_ocr.png")))
        self.assertTrue(
            os.path.exists(os.path.join("pytesseract_profile", "run_ocr.png"))
        )

        # test autosave
        for autosave_file in ("blank_card.png", "joy_of_data.png"):
            self.assertTrue(
                os.path.exists(os.path.join("pytesseract_autosave", autosave_file))
            )
            self.assertTrue(
                os.path.exists(os.path.join("easy_ocr_autosave", autosave_file))
            )
