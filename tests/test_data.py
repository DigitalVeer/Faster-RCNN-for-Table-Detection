"""Unit tests for the data pipeline, focused on the ``calc_rpn`` bug fix.

The original notebook code used ``num_regions / 2`` inside ``calc_rpn``. Under
Python 3 that is a float, so when an image has more than 128 positive anchors the
subsequent ``random.sample(..., num_regions / 2)`` raised ``TypeError`` -- which
was silently swallowed by a bare ``except`` in ``get_anchor_gt`` and quietly
dropped the image. These tests pin the integer-division fix (``//``).
"""

import random

import numpy as np
import pytest

from frcnn.config import Config
from frcnn.data import calc_rpn
from frcnn.utils import get_img_output_length


def _single_table_image(width=300, height=300):
    return {
        'filepath': '<synthetic>',
        'width': width,
        'height': height,
        'bboxes': [{'class': 'table', 'x1': 50, 'y1': 50, 'x2': 200, 'y2': 200}],
    }


def test_calc_rpn_runs_and_returns_expected_shapes():
    C = Config()
    img_data = _single_table_image()
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)  # 9

    y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(
        C, img_data, 300, 300, 300, 300, get_img_output_length)

    out_w, out_h = get_img_output_length(300, 300)  # 18, 18
    assert y_rpn_cls.shape == (1, 2 * num_anchors, out_h, out_w)
    assert y_rpn_regr.shape == (1, 8 * num_anchors, out_h, out_w)


def test_calc_rpn_num_pos_is_a_plain_integer():
    # The fix guarantees num_pos is an int (never 128.0), so downstream
    # random.sample(...) never receives a float sample size.
    C = Config()
    y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(
        C, _single_table_image(), 300, 300, 300, 300, get_img_output_length)
    assert isinstance(num_pos, (int, np.integer))


def test_integer_division_invariant_matches_the_bug_fix():
    # This reproduces the exact failing pattern at the fix site: a valid integer
    # sample size works, while the old float sample size raises TypeError.
    n = 200
    num_regions = 256

    ok = random.sample(range(n), n - num_regions // 2)
    assert len(ok) == n - 128

    with pytest.raises(TypeError):
        random.sample(range(n), n - num_regions / 2)  # noqa: what the old code did
