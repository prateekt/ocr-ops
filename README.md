# ocr-ops

OCR-Ops is infrastructure to perform Optimal Character Recognition (OCR) at scale on a large number of images and videos.
Built on top of the algo-ops framework, OCR-Ops is modular and extensible in its data processing operations.

Key Features:

* Supports building an OCRPipeline that can utilize multiple popular OCR annotation methods (e.g. PyTesseract, EasyOCR,
  etc.) and return the results in structured and efficient fashion within a unified framework.
* Enables multi-levels of information of the OCR application (e.g. text-only, bounding boxes, etc.)
* Allows definition of an image pre-processing pipeline (before OCR) and a text-cleaning pipeline (after OCR) of
  detected but noisy text to enable optimal and robust OCR performance.
* Supports several nice presets that are plug-and-play for the above purpose!