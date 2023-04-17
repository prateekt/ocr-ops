from algo_ops.ops.text import TextOp

from ocr_ops.framework.op.result.ocr_result import OCRImageResult


class DropOp(TextOp):
    """
    DropOp is used to drop intermediate results from the pipeline.
    """

    @staticmethod
    def drop_intermediate_images(result: OCRImageResult) -> OCRImageResult:
        """
        Helper function to drop intermediate images from OCRPipelineResult. This is helpful in conserving RAM when
        operating on videos.

        param result: OCRImageResult to drop intermediate images from
        """
        result.input_img.img = None
        result.output_img = None
        return result

    def __init__(self):
        super().__init__(func=self.drop_intermediate_images)
