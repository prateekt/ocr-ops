import os
import unittest

import ezplotly.settings as plot_settings
from algo_ops.dependency.tester_util import clean_paths

from ocr_ops.framework.op.ffmpeg_op import FFMPEGOp
from ocr_ops.framework.op.result.ffmeg_result import FFMPEGResult


class TestFFMPEGOp(unittest.TestCase):
    @staticmethod
    def _clean_env() -> None:
        clean_paths(dirs=("ffmpeg_op_test", "ffmpeg_op_test_fps1", "ffmpeg_profile"))

    def setUp(self) -> None:
        # paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_video = os.path.join(dir_path, "data", "test.avi")

        # env
        self._clean_env()
        plot_settings.SUPPRESS_PLOTS = True

    def tearDown(self) -> None:
        self._clean_env()

    def test_ffmpeg_op(self) -> None:
        """
        End-to-End Test.
        """

        # init FFMPEG Op
        op = FFMPEGOp(image_out_path="ffmpeg_op_test")
        self.assertEqual(op.input, None)
        self.assertEqual(op.output, None)
        self.assertEqual(op.image_out_path, "ffmpeg_op_test")
        self.assertEqual(op.fps, 10)
        self.assertEqual(op.eval_func, None)
        self.assertEqual(len(op.execution_times), 0)
        self.assertEqual(op.incorrect_pkl_path, None)
        for method in (op.vis, op.vis_profile, op.save_input, op.save_output):
            with self.assertRaises(ValueError):
                method()

        # run video through op (fps=10)
        op.exec(inp=self.test_video)
        self.assertTrue(isinstance(op.input, str))
        self.assertTrue(isinstance(op.output, FFMPEGResult))
        self.assertEqual(op.input, self.test_video)
        self.assertEqual(op.output.input_video_path, self.test_video)
        self.assertEqual(op.output.output_images_path, "ffmpeg_op_test")
        self.assertEqual(op.output.fps, 10)
        self.assertEqual(op.image_out_path, "ffmpeg_op_test")
        self.assertEqual(op.fps, 10)
        self.assertEqual(op.eval_func, None)
        self.assertEqual(len(op.execution_times), 1)
        self.assertEqual(op.incorrect_pkl_path, None)
        self.assertEqual(len(os.listdir("ffmpeg_op_test")), 30)

        # run video again at different fps (fps=1) and check state
        op.fps = 1
        op.image_out_path = "ffmpeg_op_test_fps1"
        op.exec(inp=self.test_video)
        self.assertTrue(isinstance(op.input, str))
        self.assertTrue(isinstance(op.output, FFMPEGResult))
        self.assertEqual(op.input, self.test_video)
        self.assertEqual(op.output.input_video_path, self.test_video)
        self.assertEqual(op.output.output_images_path, "ffmpeg_op_test_fps1")
        self.assertEqual(op.output.fps, 1)
        self.assertEqual(op.image_out_path, "ffmpeg_op_test_fps1")
        self.assertEqual(op.fps, 1)
        self.assertEqual(op.eval_func, None)
        self.assertEqual(len(op.execution_times), 2)
        self.assertEqual(op.incorrect_pkl_path, None)
        self.assertEqual(len(os.listdir("ffmpeg_op_test_fps1")), 3)

        # test visualization
        op.vis_input()
        op.vis()
        op.save_input()
        op.save_output()

        # test profile
        op.vis_profile(profiling_figs_path="ffmpeg_profile")
        self.assertTrue(
            os.path.exists(
                os.path.join("ffmpeg_profile", "_convert_to_images_wrapper.png")
            )
        )
