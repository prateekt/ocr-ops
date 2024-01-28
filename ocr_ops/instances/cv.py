import cv2
import numpy as np
from algo_ops.ops.cv import ImageResult
from algo_ops.pipeline.cv_pipeline import CVPipeline


def _gray_scale(inp: ImageResult) -> ImageResult:
    # convert to gray scale
    img = inp.img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ImageResult(img=gray, file_path=inp.file_path, cmap="gray")


def _invert_black_channel(inp: ImageResult) -> ImageResult:
    # extract black channel in CMYK color space
    # (after this transformation, it appears white)
    img = inp.img
    img_float = img.astype(float) / 255.0
    k_channel = 1 - np.max(img_float, axis=2)
    k_channel = (255 * k_channel).astype(np.uint8)
    return ImageResult(img=k_channel, file_path=inp.file_path, cmap="gray")


def _remove_background(inp: ImageResult, lower_lim: int = 190) -> ImageResult:
    # remove background that is not white
    img = inp.img
    _, bin_img = cv2.threshold(img, lower_lim, 255, cv2.THRESH_BINARY)
    return ImageResult(img=bin_img, file_path=inp.file_path, cmap="gray")


def _invert_back(inp: ImageResult) -> ImageResult:
    # Invert back to black text / white background
    img = inp.img
    inv_img = cv2.bitwise_not(img)
    return ImageResult(img=inv_img, file_path=inp.file_path, cmap="gray")


def basic_cv_pipeline() -> CVPipeline:
    """
    Just gray scale image.
    """

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_gray_scale],
    )
    return img_pipeline


def black_text_cv_pipeline() -> CVPipeline:
    """
    Initializes computer vision pipeline to isolate black text in an image.
    """

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_invert_black_channel, _remove_background, _invert_back],
    )
    return img_pipeline


def white_text_cv_pipeline() -> CVPipeline:
    """
    Initializes computer vision pipeline to isolate white text in an image.
    """

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_gray_scale, _remove_background, _invert_back],
    )
    return img_pipeline
