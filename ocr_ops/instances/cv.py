import cv2
import numpy as np
from algo_ops.pipeline.cv_pipeline import CVPipeline


def _gray_scale(img: np.array) -> np.array:
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def _invert_black_channel(img: np.array) -> np.array:
    # extract black channel in CMYK color space
    # (after this transformation, it appears white)
    img_float = img.astype(float) / 255.0
    k_channel = 1 - np.max(img_float, axis=2)
    k_channel = (255 * k_channel).astype(np.uint8)
    return k_channel


def _remove_background(img: np.array, lower_lim: int = 190) -> np.array:
    # remove background that is not white
    _, bin_img = cv2.threshold(img, lower_lim, 255, cv2.THRESH_BINARY)
    return bin_img


def _invert_back(img: np.array) -> np.array:
    # Invert back to black text / white background
    inv_img = cv2.bitwise_not(img)
    return inv_img


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
