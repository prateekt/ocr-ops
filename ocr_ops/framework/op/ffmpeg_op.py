from typing import Optional

from algo_ops.ops.op import Op

from ocr_ops.dependency.ffmpeg import FFMPEG


class FFMPEGOp(Op):
    """
    FFMPEGOP is used to convert a video into image frames stored in a directory. It turns the use of FFMPEG video ->
    frames conversion into an Op that can placed into an OCR pipeline.
    """

    def _convert_to_images_wrapper(self, video_path: str) -> str:
        """
        Wrapper function to convert a video into image frames.

        param video_path: Path to video file

        Return:
            images_frame_path: Path to directory containing frame images extracted from video using FFMPEG.
        """
        success, image_frames_path = FFMPEG.convert_video_to_frames(
            video_path=video_path, out_path=self.out_path, fps=self.fps
        )
        if not success:
            raise SystemError("FFMPEG conversion failed on " + str(video_path))
        return image_frames_path

    def __init__(self, out_path: Optional[str] = None, fps: int = 10):
        """
        :param out_path: Path to output directory where images should be extracted
        :param fps: Frame per second
        """
        self.out_path: str = out_path
        self.fps: int = fps
        super().__init__(func=self._convert_to_images_wrapper)

    def vis(self) -> None:
        if self.input is None:
            raise ValueError("FFMPEGOp has no input.")
        if self.output is None:
            raise ValueError("FFMPEGOp has no output.")
        print("FFMEGOp Extracted " + str(self.input) + " to " + str(self.out_path))

    def vis_input(self) -> None:
        if self.input is None:
            raise ValueError("FFMPEGOp has no input.")
        print("FFMEGOp Input: " + str(self.input))

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        self.vis_input()

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        if self.output is None:
            raise ValueError("FFMPEGOp has no output.")
        print("FFMEGOp Output: " + str(self.output))
