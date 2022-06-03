import os
import unittest
from typing import Callable

import ezplotly.settings as plot_settings
import numpy as np
from algo_ops.dependency.tester_util import iter_params, clean_paths
from shapely.geometry import Polygon

from ocr_ops.framework.op.ocr_op import (
    PyTesseractTextOCROp,
    PyTesseractTextBoxOCROp,
    EasyOCRTextOp,
    EasyOCRTextBoxOp,
)
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
    @staticmethod
    def _clean_env() -> None:
        clean_paths(
            dirs=(
                "ocr_out",
                "algo_ops_profile",
                "pytesseract_autosave",
                "easyocr_autosave",
                "general_autosave",
                "test_figs",
            ),
            files=("test.pkl",),
        )

    def _assert_pass_vis_tests(self, ocr_pipeline: OCRPipeline) -> None:
        # test vis input / output
        with self.assertRaises(ValueError):
            ocr_pipeline.vis_input()
        ocr_pipeline.vis()

        # test vis profile
        ocr_pipeline.vis_profile()
        for file in ["['exec_ocr']", "['exec_ocr']_violin", "exec_ocr"]:
            self.assertTrue(
                os.path.exists(os.path.join("algo_ops_profile", file + ".png"))
            )

    def _assert_pass_save_tests(
        self, ocr_pipeline: OCRPipeline, expected_num_txt: int, expected_num_png: int
    ) -> None:
        # test save input / output
        with self.assertRaises(ValueError):
            ocr_pipeline.save_input()
        ocr_pipeline.save_output("ocr_out")
        self.assertEqual(
            len([a for a in os.listdir("ocr_out") if ".txt" in a]), expected_num_txt
        )
        self.assertEqual(
            len([a for a in os.listdir("ocr_out") if ".png" in a]), expected_num_png
        )

    def setUp(self) -> None:
        # paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.joy_of_data_img = os.path.join(dir_path, "data", "joy_of_data.png")

        # env
        self._clean_env()
        plot_settings.SUPPRESS_PLOTS = True

    def tearDown(self) -> None:
        self._clean_env()

    def test_ocr_pytesseract_text_pipeline(self) -> None:
        """
        Test PyTesseract OCR text-only pipeline.
        """

        # init and check state
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.PYTESSERACT,
            output_type=OutputType.TEXT,
            text_pipeline=None,
            autosave_img_path="pytesseract_autosave",
        )
        self.assertTrue(isinstance(ocr_pipeline.ocr_op, PyTesseractTextOCROp))
        self.assertEqual(ocr_pipeline.input, None)
        self.assertEqual(ocr_pipeline.output, None)
        for method in [
            ocr_pipeline.vis,
            ocr_pipeline.vis_profile,
            ocr_pipeline.save_input,
            ocr_pipeline.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()

        # test execution on sample image
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, OCRResult))
        self.assertFalse(output.use_bounding_box)
        self.assertTrue(np.array_equal(output.input_img, output.output_img))
        self.assertEqual(len(output), 1)
        self.assertTrue(isinstance(output[0], TextBox))
        self.assertEqual(output[0].text, "joy of data\n")
        self.assertEqual(output[0].bounding_box, None)
        self.assertEqual(output[0].conf, None)

        # test save and vis
        self._assert_pass_save_tests(
            ocr_pipeline=ocr_pipeline, expected_num_txt=1, expected_num_png=1
        )
        self._assert_pass_vis_tests(ocr_pipeline=ocr_pipeline)

        # test autosave
        self.assertTrue(
            os.path.exists(os.path.join("pytesseract_autosave", "joy_of_data.txt"))
        )

        # test pickle pipeline
        ocr_pipeline.to_pickle(out_pkl_path="test.pkl")

    def test_ocr_pytesseract_textbox_pipeline(self) -> None:
        """
        Test PyTesseract OCR TextBox pipeline.
        """

        # init and check state
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.PYTESSERACT,
            output_type=OutputType.TEXTBOX,
            text_pipeline=None,
            autosave_img_path="pytesseract_autosave",
        )
        self.assertTrue(isinstance(ocr_pipeline.ocr_op, PyTesseractTextBoxOCROp))
        self.assertEqual(ocr_pipeline.input, None)
        self.assertEqual(ocr_pipeline.output, None)
        for method in [
            ocr_pipeline.vis,
            ocr_pipeline.vis_profile,
            ocr_pipeline.save_input,
            ocr_pipeline.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()

        # test execution on sample image
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

        # test save  and vis
        self._assert_pass_save_tests(
            ocr_pipeline=ocr_pipeline, expected_num_txt=0, expected_num_png=2
        )
        self._assert_pass_vis_tests(ocr_pipeline=ocr_pipeline)

        # test autosave
        self.assertTrue(
            os.path.exists(os.path.join("pytesseract_autosave", "joy_of_data.png"))
        )

        # test pickle
        ocr_pipeline.to_pickle(out_pkl_path="test.pkl")

    def test_easyocr_text_pipeline(self) -> None:
        """
        Test EasyOCR text-only pipeline.
        """

        # init and check state
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.EASYOCR,
            output_type=OutputType.TEXT,
            text_pipeline=None,
            autosave_img_path="easyocr_autosave",
        )
        self.assertTrue(isinstance(ocr_pipeline.ocr_op, EasyOCRTextOp))
        self.assertEqual(ocr_pipeline.input, None)
        self.assertEqual(ocr_pipeline.output, None)
        for method in (
            ocr_pipeline.vis,
            ocr_pipeline.vis_profile,
            ocr_pipeline.save_input,
            ocr_pipeline.save_output,
        ):
            with self.assertRaises(ValueError):
                method()

        # test execution on sample image
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

        # test save  and vis
        self._assert_pass_save_tests(
            ocr_pipeline=ocr_pipeline, expected_num_txt=1, expected_num_png=1
        )
        self._assert_pass_vis_tests(ocr_pipeline=ocr_pipeline)

        # test autosave
        self.assertTrue(
            os.path.exists(os.path.join("easyocr_autosave", "joy_of_data.txt"))
        )

        # test pickle
        ocr_pipeline.to_pickle(out_pkl_path="test.pkl")

    def test_easyocr_textbox_pipeline(self) -> None:
        """
        Test EasyOCR TextBox pipeline.
        """

        # init and check state
        ocr_pipeline = OCRPipeline(
            img_pipeline=None,
            ocr_method=OCRMethod.EASYOCR,
            output_type=OutputType.TEXTBOX,
            text_pipeline=None,
            autosave_img_path="easyocr_autosave",
        )
        self.assertTrue(isinstance(ocr_pipeline.ocr_op, EasyOCRTextBoxOp))
        self.assertEqual(ocr_pipeline.input, None)
        self.assertEqual(ocr_pipeline.output, None)
        for method in [
            ocr_pipeline.vis,
            ocr_pipeline.vis_profile,
            ocr_pipeline.save_input,
            ocr_pipeline.save_output,
        ]:
            with self.assertRaises(ValueError):
                method()

        # test execution on sample image
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

        # test save and vis
        self._assert_pass_save_tests(
            ocr_pipeline=ocr_pipeline, expected_num_txt=0, expected_num_png=2
        )
        self._assert_pass_vis_tests(ocr_pipeline=ocr_pipeline)

        # test autosave
        self.assertTrue(
            os.path.exists(os.path.join("easyocr_autosave", "joy_of_data.png"))
        )

        # test pickle
        ocr_pipeline.to_pickle(out_pkl_path="test.pkl")

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
            autosave_img_path="general_autosave",
        )
        ocr_pipeline.set_text_pipeline_params("_check_vocab", {"vocab_words": {"joy"}})
        output = ocr_pipeline.exec(self.joy_of_data_img)
        self.assertListEqual(output.words, ["joy"])

        # test autosave
        self.assertTrue(
            os.path.exists(os.path.join("general_autosave", "joy_of_data.txt"))
            or os.path.exists(os.path.join("general_autosave", "joy_of_data.png"))
        )

        # test pickle
        ocr_pipeline.to_pickle(out_pkl_path="test.pkl")

    @iter_params(pipeline_init=(black_text_cv_pipeline, white_text_cv_pipeline))
    def test_cvpipeline_instances(self, pipeline_init: Callable) -> None:
        """
        Test CVPipeline instances.
        """
        # run pipeline
        cv_pipeline = pipeline_init()
        output = cv_pipeline.exec(self.joy_of_data_img)
        self.assertTrue(isinstance(output, np.ndarray))

        # attempt vis
        with self.assertRaises(ValueError):
            cv_pipeline.vis_input()
        cv_pipeline.vis()
        cv_pipeline.vis_profile()

        # attempt save
        with self.assertRaises(ValueError):
            cv_pipeline.save_input()
        cv_pipeline.save_output(out_path="test_figs")
        self.assertTrue(len(os.listdir("test_figs")), 8)

        # test pickle
        cv_pipeline.to_pickle(out_pkl_path="test.pkl")

    @iter_params(pipeline_init=(basic_text_cleaning_pipeline,))
    def test_textpipeline_instances(self, pipeline_init: Callable) -> None:
        """
        Test text cleaning pipeline instances.
        """

        # basic cleaning pipeline
        text_pipeline = pipeline_init()
        text_pipeline.set_pipeline_params(
            "_check_vocab", {"vocab_words": {"joy", "of", "data"}}
        )
        output = text_pipeline.exec("joy of ***%$## data opfu")
        self.assertListEqual(output, ["joy", "of", "data"])
        output = text_pipeline.exec(["joy of   \n", "***%$## \n" "data\n opfu\n\n\n\n"])
        self.assertListEqual(output, ["joy", "of", "data"])
        text_pipeline.set_pipeline_params("_check_vocab", {"vocab_words": {"data"}})
        output = text_pipeline.exec(["joy of   \n", "***%$## \n" "data\n opfu\n\n\n\n"])
        self.assertListEqual(output, ["data"])

        # test pickle
        text_pipeline.to_pickle(out_pkl_path="test.pkl")

    def test_ocr_pipeline_instances(self) -> None:
        """
        Test all OCR pipeline instances.
        """

        # basic pipeline
        p1 = basic_ocr_pipeline()
        output = p1.exec(inp=self.joy_of_data_img)
        self.assertEqual(output[0].text.strip(), "joy of data")
        p1.to_pickle("test.pkl")

        # basic pipeline with text cleaning
        p2 = basic_ocr_with_text_cleaning_pipeline(vocab_words={"joy", "of"})
        output = p2.exec(inp=self.joy_of_data_img)
        self.assertListEqual(output.words, ["joy", "of"])
        p2.to_pickle("test.pkl")

        # black text ocr
        p3 = black_text_ocr_pipeline()
        p3.set_text_pipeline_params(
            "_check_vocab", {"vocab_words": {"joy", "data", "of"}}
        )
        output = p3.exec(inp=self.joy_of_data_img)
        self.assertEqual(output.words, [])
        p3.to_pickle("test.pkl")

        # white text ocr
        p4 = white_text_ocr_pipeline()
        p4.set_text_pipeline_params("_check_vocab", {"vocab_words": {"data"}})
        output = p4.exec(inp=self.joy_of_data_img)
        self.assertListEqual(output.words, ["data"])
        p4.to_pickle("test.pkl")
