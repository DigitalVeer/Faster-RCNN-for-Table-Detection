"""Unit tests for the pure-geometry ROI helpers (NumPy only, no TensorFlow)."""

import numpy as np
import pytest

from frcnn.roi import (
    apply_regr,
    apply_regr_np,
    intersection,
    iou,
    non_max_suppression_fast,
    union,
)


def test_iou_identical_boxes():
    box = [0, 0, 10, 10]
    assert iou(box, box) == pytest.approx(1.0, abs=1e-4)


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 10x5 strip:
    # intersection = 50, union = 100 + 100 - 50 = 150 -> 1/3
    a = [0, 0, 10, 10]
    b = [0, 5, 10, 15]
    assert iou(a, b) == pytest.approx(50.0 / 150.0, abs=1e-4)


def test_intersection_and_union():
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    inter = intersection(a, b)
    assert inter == 25
    assert union(a, b, inter) == 100 + 100 - 25


def test_intersection_no_overlap_is_zero():
    assert intersection([0, 0, 5, 5], [10, 10, 20, 20]) == 0


def test_nms_keeps_highest_prob_and_suppresses_overlap():
    # Two near-identical boxes; NMS should keep only the higher-probability one.
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    probs = np.array([0.9, 0.6])
    kept_boxes, kept_probs = non_max_suppression_fast(boxes, probs, overlap_thresh=0.5)
    assert kept_boxes.shape[0] == 1
    assert kept_probs[0] == pytest.approx(0.9)


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    probs = np.array([0.8, 0.7])
    kept_boxes, _ = non_max_suppression_fast(boxes, probs, overlap_thresh=0.5)
    assert kept_boxes.shape[0] == 2


def test_nms_respects_max_boxes():
    boxes = np.array([[i, i, i + 5, i + 5] for i in range(0, 200, 10)], dtype=float)
    probs = np.linspace(0.1, 0.9, boxes.shape[0])
    kept_boxes, _ = non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=3)
    assert kept_boxes.shape[0] == 3


def test_apply_regr_identity_with_zero_deltas():
    # Zero regression deltas must leave (x, y, w, h) unchanged.
    assert apply_regr(10, 20, 30, 40, 0.0, 0.0, 0.0, 0.0) == (10, 20, 30, 40)


def test_apply_regr_np_identity_with_zero_deltas():
    X = np.array([
        [[10.0]],
        [[20.0]],
        [[30.0]],
        [[40.0]],
    ])
    T = np.zeros_like(X)
    out = apply_regr_np(X, T)
    np.testing.assert_allclose(out.reshape(4), [10, 20, 30, 40])
