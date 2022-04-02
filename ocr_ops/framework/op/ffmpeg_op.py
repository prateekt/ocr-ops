from typing import Optional

from algo_ops.ops.op import Op

from ocr_ops.dependency.ffmpeg import FFMPEG


class FFMPEGOp(Op):
    """
    Turn the use of FFMPEG video -> frames conversion into an Op that can placed into an OCR pipeline.
    """

    def _convert_to_images_wrapper(self, video_path: str):
        """
        Wrapper function to convert a video into image frames.

        param video_path: Path to video file

        Return:
            images_frame_path: Path to directory containing frame images extracted from video using FFMPEG.

        """
        success, image_frames_path = FFMPEG.convert_video_to_frames(
            video_path=video_path, out_path=self.out_path
        )
        if not success:
            raise SystemError("FFMPEG conversion failed on " + str(video_path))
        return image_frames_path

    def __init__(self, out_path: Optional[str] = None):
        self.out_path = out_path
        super().__init__(func=self._convert_to_images_wrapper)

    def vis(self) -> None:
        print("Converted " + str(self.input) + ".")

    def vis_input(self) -> None:
        pass

    def save_input(self, out_path: str) -> None:
        pass

    def save_output(self, out_path) -> None:
        pass
