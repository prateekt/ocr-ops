import os
import shutil
import unittest

import algo_ops.plot.settings as plot_settings

from ocr_ops.framework.op.ffmpeg_op import FFMPEGOp


class TestFFMPEGOp(unittest.TestCase):
    def _clean_env(self) -> None:
        if os.path.exists("ffmpeg_op_test"):
            shutil.rmtree("ffmpeg_op_test")
        if os.path.exists("ffmpeg_op_test_fps1"):
            shutil.rmtree("ffmpeg_op_test_fps1")
        if os.path.exists("ffmpeg_profile"):
            shutil.rmtree("ffmpeg_profile")

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
        op = FFMPEGOp(out_path="ffmpeg_op_test")
        self.assertEqual(op.input, None)
        self.assertEqual(op.output, None)
        self.assertEqual(op.out_path, "ffmpeg_op_test")
        self.assertEqual(op.fps, 10)
        self.assertEqual(op.eval_func, None)
        self.assertEqual(len(op.execution_times), 0)
        self.assertEqual(op.incorrect_pkl_path, None)
        for method in (op.vis, op.vis_profile, op.save_input, op.save_output):
            with self.assertRaises(ValueError):
                method()

        # run video through op (fps=10)
        op.exec(inp=self.test_video)
        self.assertEqual(op.input, self.test_video)
        self.assertEqual(op.output, "ffmpeg_op_test")
        self.assertEqual(op.out_path, "ffmpeg_op_test")
        self.assertEqual(op.fps, 10)
        self.assertEqual(op.eval_func, None)
        self.assertEqual(len(op.execution_times), 1)
        self.assertEqual(op.incorrect_pkl_path, None)
        self.assertEqual(len(os.listdir("ffmpeg_op_test")), 30)

        # run video again at different fps (fps=1) and check state
        op.fps = 1
        op.out_path = "ffmpeg_op_test_fps1"
        op.exec(inp=self.test_video)
        self.assertEqual(op.input, self.test_video)
        self.assertEqual(op.output, "ffmpeg_op_test_fps1")
        self.assertEqual(op.out_path, "ffmpeg_op_test_fps1")
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
