"""Faster R-CNN for Table Detection - shared library code.

This package holds the model-independent building blocks that were previously
copy-pasted between ``1_train_model.ipynb`` and ``2_test_model.ipynb``. Keeping a
single, importable copy removes ~1000 lines of duplication and lets the pure
functions be unit tested.

Only framework-independent code (NumPy / OpenCV) lives here so far:

- :mod:`frcnn.config`  -- the ``Config`` hyper-parameter container
- :mod:`frcnn.utils`   -- image-size / feature-map helpers
- :mod:`frcnn.roi`     -- IoU, non-max suppression, ROI generation
- :mod:`frcnn.data`    -- annotation parsing, augmentation, RPN targets
- :mod:`frcnn.eval`    -- image formatting and mean-average-precision

The Keras/TensorFlow model and loss definitions (``nn_base``, ``rpn_layer``,
``classifier_layer``, ``RoiPoolingConv`` and the loss functions) are migrated to
``tensorflow.keras`` separately, so importing this package does not require
TensorFlow.
"""

from .config import Config

__all__ = ["Config"]
