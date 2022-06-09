class FFMPEGResult:
    """
    Represents the result of FFMPEGOp.
    """

    def __init__(self, input_video_path: str, output_images_path: str, fps: int):
        """
        param input_video_path: Path to input video file
        param output_images_path: Path to where extracted images are stored
        param fps: The frame rate to extract at
        """
        self.input_video_path: str = input_video_path
        self.output_images_path: str = output_images_path
        self.fps: int = fps

    def vis(self) -> None:
        print(
            "FFMEGOp extracted "
            + str(self.input_video_path)
            + " to "
            + str(self.output_images_path)
            + " at "
            + str(self.fps)
            + "fps."
        )
