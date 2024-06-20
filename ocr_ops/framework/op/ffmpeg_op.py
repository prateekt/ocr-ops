from typing import Optional

from algo_ops.ops.op import Op

from ocr_ops.dependency.ffmpeg import FFMPEG
from ocr_ops.framework.op.result.ffmeg_result import FFMPEGResult, FFMPEGMergeResult


class FFMPEGOp(Op):
    """
    FFMPEGOp is used to convert a video into image frames stored in a directory. It turns the use of FFMPEG video ->
    frames conversion into an Op that can placed into an OCR pipeline.
    """

    def _convert_to_images_wrapper(self, video_path: str) -> FFMPEGResult:
        """
        Wrapper function to convert a video into image frames.

        param video_path: Path to video file

        Return:
            images_frame_path: Path to directory containing frame images extracted from video using FFMPEG.
        """
        if self.image_out_path is None:
            raise ValueError("No out_video_path specified.")
        success, image_frames_path = FFMPEG.convert_video_to_frames(
            video_path=video_path, out_path=self.image_out_path, fps=self.fps
        )
        if not success:
            raise SystemError("FFMPEG conversion failed on " + str(video_path))
        result = FFMPEGResult(
            input_video_path=video_path,
            output_images_path=image_frames_path,
            fps=self.fps,
        )
        return result

    def __init__(self, image_out_path: Optional[str] = None, fps: int = 10):
        """
        param out_video_path: Path to output directory where images should be extracted
        param fps: Frame per second
        """
        self.image_out_path: str = image_out_path
        self.fps: int = fps
        super().__init__(func=self._convert_to_images_wrapper)
        self.input: Optional[str] = None
        self.output: Optional[FFMPEGResult] = None

    def vis(self) -> None:
        if self.input is None:
            raise ValueError("FFMPEGOp has no input video path.")
        if self.output is None:
            raise ValueError("FFMPEGOp has not produced output.")
        self.output.vis()

    def vis_input(self) -> None:
        if self.input is None:
            raise ValueError("FFMPEGOp has no input video path.")
        print("FFMEGOp Input: " + str(self.input))

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        self.vis_input()

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        self.vis()


class FFMPEGMergeOp(Op):
    """
    FFMPEGMergeOp is used to convert image frames stored in a directory to video.
    It turns the use of FFMPEG frames -> video conversion into an Op that can placed into an OCR pipeline.
    """

    def _convert_to_video_wrapper(self, frames_path: str) -> FFMPEGMergeResult:
        """
        Wrapper function to convert image frames into video.

        param frames_path: Path to image frames directory

        Return:
            video_path: Path to video file created from image frames using FFMPEG.
        """
        if self.out_video_path is None:
            raise ValueError("No output video path specified.")
        success, out_video_path = FFMPEG.convert_frames_to_video(
            frames_path=frames_path, out_video_path=self.out_video_path, fps=self.fps
        )
        if not success:
            raise SystemError("FFMPEG conversion failed on " + str(frames_path))
        result = FFMPEGMergeResult(
            input_frames_path=frames_path,
            output_video_path=out_video_path,
            fps=self.fps,
        )
        return result

    def __init__(self, out_video_path: Optional[str] = None, fps: int = 10):
        """
        param out_video_path: Path to where video should be saved
        param fps: Frame per second
        """
        self.out_video_path: str = out_video_path
        self.fps: int = fps
        super().__init__(func=self._convert_to_video_wrapper)
        self.input: Optional[str] = None
        self.output: Optional[FFMPEGMergeResult] = None

    def vis(self) -> None:
        if self.input is None:
            raise ValueError("FFMPEGOp has no input frames path.")
        if self.output is None:
            raise ValueError("FFMPEGOp has not produced output.")
        self.output.vis()

    def vis_input(self) -> None:
        if self.input is None:
            raise ValueError("FFMPEGOp has no input frames path.")
        print("FFMEGOp Input: " + str(self.input))

    def save_input(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        self.vis_input()

    def save_output(self, out_path: str = ".", basename: Optional[str] = None) -> None:
        self.vis()
