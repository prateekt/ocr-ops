import os
import shutil
import unittest

from ocr_ops.framework.op.abstract_ocr_op import OCRResult
from ocr_ops.framework.op.ocr_op import (
    EasyOCRTextOp,
    PyTesseractTextOCROp,
    EasyOCRTextBoxOp,
    PyTesseractTextBoxOCROp,
)


class TestOCROps(unittest.TestCase):
    @staticmethod
    def _clean_env():
        if os.path.exists("txt_ocr_output"):
            shutil.rmtree("txt_ocr_output")
        if os.path.exists("easy_ocr_profile"):
            shutil.rmtree("easy_ocr_profile")
        if os.path.exists("pytesseract_profile"):
            shutil.rmtree("pytesseract_profile")
        if os.path.exists("box_ocr_output"):
            shutil.rmtree("box_ocr_output")

    def setUp(self) -> None:

        # paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.joy_of_data_img = os.path.join(dir_path, "data", "joy_of_data.png")
        self.blank_card_img = os.path.join(dir_path, "data", "blank_card.png")
        self._clean_env()

    def tearDown(self) -> None:
        self._clean_env()

    def test_text_ocr(self) -> None:
        """
        Test TextOCROp on test images.
        """

        # init
        easy_ocr_op = EasyOCRTextOp()
        pytesseract_op = PyTesseractTextOCROp()

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
        self.assertTrue(isinstance(output, OCRResult))
        self.assertEqual(output.to_text_list(), ["joy", "of", "data"])
        output = easy_ocr_op.exec(self.blank_card_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertEqual(output.to_text_list(), [])

        # test pytesseract on test images
        output = pytesseract_op.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertEqual(output.to_text_list(), ["joy of data\n"])
        output = pytesseract_op.exec(self.blank_card_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertEqual(output.to_text_list(), [" \n\n \n"])

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
        shutil.rmtree("txt_ocr_output")

        # test visualizing profile
        easy_ocr_op.vis_profile(profiling_figs_path="easy_ocr_profile")
        pytesseract_op.vis_profile(profiling_figs_path="pytesseract_profile")
        self.assertTrue(
            os.path.exists(os.path.join("easy_ocr_profile", "exec_ocr.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join("pytesseract_profile", "exec_ocr.png"))
        )
        shutil.rmtree("easy_ocr_profile")
        shutil.rmtree("pytesseract_profile")

    def test_spatial_ocr_op(self) -> None:
        """
        Test SpatialOCROp on test images.
        """

        # init
        easy_ocr_op = EasyOCRTextBoxOp()
        pytesseract_op = PyTesseractTextBoxOCROp()

        # test that ops without inputs don't do much
        self.assertEqual(easy_ocr_op.input, None)
        self.assertEqual(pytesseract_op.input, None)
        self.assertEqual(easy_ocr_op.output, None)
        self.assertEqual(pytesseract_op.output, None)
        self.assertEqual(len(easy_ocr_op.execution_times), 0)
        self.assertEqual(len(pytesseract_op.execution_times), 0)
        for method in [
            easy_ocr_op.vis_input,
            easy_ocr_op.vis,
            easy_ocr_op.vis_profile,
            easy_ocr_op.save_input,
            easy_ocr_op.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()
        for method in [
            pytesseract_op.vis_input,
            pytesseract_op.vis,
            pytesseract_op.vis_profile,
            pytesseract_op.save_input,
            pytesseract_op.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()

        # test that spatial bounding boxes overlap significantly between the two OCR methods for the same detected
        # text on joy of data image
        output1: OCRResult = easy_ocr_op.exec(self.joy_of_data_img)
        output2: OCRResult = pytesseract_op.exec(self.joy_of_data_img)
        output2.text_boxes = [
            output
            for output in output2
            if output.conf != -1.0 and len(output.text.strip()) > 0
        ]
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
        shutil.rmtree("box_ocr_output")

        # test that nothing is detected in blank image
        output1: OCRResult = easy_ocr_op.exec(self.blank_card_img)
        output2: OCRResult = pytesseract_op.exec(self.blank_card_img)
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
        self.assertTrue(
            os.path.exists(os.path.join("easy_ocr_profile", "exec_ocr.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join("pytesseract_profile", "exec_ocr.png"))
        )
        shutil.rmtree("easy_ocr_profile")
        shutil.rmtree("pytesseract_profile")
