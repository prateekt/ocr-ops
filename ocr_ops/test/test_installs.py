import os
import unittest

import easyocr
import pytesseract
from algo_ops.dependency.tester_util import clean_paths

from ocr_ops.dependency.ffmpeg import FFMPEG


class TestOCRDependencies(unittest.TestCase):
    def setUp(self) -> None:
        # paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_image1 = os.path.join(dir_path, "data", "joy_of_data.png")
        self.test_image2 = os.path.join(dir_path, "data", "blank_card.png")
        self.test_video = os.path.join(dir_path, "data", "test.avi")

    def test_text_easy_ocr(self) -> None:
        """
        Test EasyOCr on sample images.
        """

        # init easy ocr reader
        easy_ocr_reader = easyocr.Reader(["en"])

        # test joy of data image
        output = easy_ocr_reader.readtext(self.test_image1, detail=0)
        self.assertListEqual(output, ["joy", "of", "data"])

        # test blank card image
        output = easy_ocr_reader.readtext(self.test_image2, detail=0)
        self.assertListEqual(output, [])

    def test_pytesseract(self) -> None:
        """
        Test PyTesseract on sample images.
        """

        # test joy of data image
        output = pytesseract.image_to_string(self.test_image1)
        self.assertEqual(output.strip(), "joy of data")

        # test blank card image
        output = pytesseract.image_to_string(self.test_image2)
        self.assertEqual(output.strip(), "")

    def test_ffmpeg(self) -> None:
        """
        Test FFMPEG conversion.
        """

        # check that FFMPEG is installed.
        self.assertTrue(FFMPEG.is_installed())

        # test video conversion (should produce 30 frames)
        FFMPEG.convert_video_to_frames(
            video_path=self.test_video, out_path="test_ffmpeg"
        )
        self.assertEqual(len(os.listdir("test_ffmpeg")), 30)
        clean_paths(dirs=("test_ffmpeg",))
