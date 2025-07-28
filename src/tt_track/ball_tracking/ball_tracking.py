"""
This module implements a ball tracking algorithm using YOLO and OpenCV.
"""

import math
import pathlib

import cv2
import numpy as np

from matplotlib.path import Path

from ultralytics import YOLO
from src.tt_track.constants import TABLE_POLYGON
from src.tt_track.constants import MAX_PIXEL_JUMP

TABLE_PATH = Path(TABLE_POLYGON)


def detect_bounces(
    records, min_movement=2.0, min_vertical_change=1.0
):
    """
    Detect ball bounces by checking for a V-shape where the middle point is below its neighbors
    and within the defined table area.

    :param records: List of dicts with 'frame', 'cx', 'cy'.
    :param min_movement: Minimum total movement to avoid noise.
    :param min_vertical_change: Minimum Y-change to qualify as a bounce.
    :return: List of frame indices where bounces are detected.
    """
    if len(records) < 3:
        return []

    collisions = []

    for i in range(1, len(records) - 1):
        prev = records[i - 1]
        curr = records[i]
        next_ = records[i + 1]

        v_in = np.array([curr["cx"] - prev["cx"], curr["cy"] - prev["cy"]])
        v_out = np.array([next_["cx"] - curr["cx"], next_["cy"] - curr["cy"]])

        if np.linalg.norm(v_in) < min_movement or np.linalg.norm(v_out) < min_movement:
            continue

        if not (curr["cy"] > prev["cy"] and curr["cy"] > next_["cy"]):
            continue

        if abs(v_in[1]) < min_vertical_change or abs(v_out[1]) < min_vertical_change:
            continue


        point = (curr["cx"], curr["cy"])
        if not TABLE_PATH.contains_point(point):
            continue

        collisions.append(curr.copy())

    return collisions


def linear_fill_trajectory(records):
    """
    Fill missing ball positions using simple linear interpolation.
    Gaps larger than 4 frames are skipped (not filled).

    :param records: List of dicts with 'frame', 'cx', 'cy' (and optionally other keys like 'x1', 'y1', 'x2', 'y2', 'confidence').
    :return: List of dicts, one per frame, with estimated 'cx', 'cy' and other keys, filled similarly to the process_video output.
    """
    if not records:
        return []

    # This probably isn't necessary, but sometimes I get a bit paranoid, so I sort the list.
    # It will appear multiple times, but the list isn't that big, so it’s not a big deal.
    detections = {r["frame"]: r for r in records}
    frame_nums = sorted(detections)

    filled_records = []
    prev = None

    for f in range(frame_nums[0], frame_nums[-1] + 1):
        if f in detections:
            filled = detections[f].copy()
            filled_records.append(filled)
            prev = (f, (filled["cx"], filled["cy"]))
        elif prev is not None:
            prev_frame, (prev_cx, prev_cy) = prev
            next_frame = next((fn for fn in frame_nums if fn > f), None)
            if next_frame is None:
                break

            if (next_frame - prev_frame - 1) <= 4:
                next_cx = detections[next_frame]["cx"]
                next_cy = detections[next_frame]["cy"]
                cx = prev_cx + ((f - prev_frame) / (next_frame - prev_frame)) * (
                    next_cx - prev_cx
                )
                cy = prev_cy + ((f - prev_frame) / (next_frame - prev_frame)) * (
                    next_cy - prev_cy
                )
                filled_records.append(
                    {
                        "frame": f,
                        "cx": float(cx),
                        "cy": float(cy),
                        "x1": int(cx - 10),
                        "y1": int(cy - 10),
                        "x2": int(cx + 10),
                        "y2": int(cy + 10),
                        "confidence": 0.0,
                    }
                )

    return filled_records

# Was forced to split into multiple functions, because of PEP8
def load_model_and_video(model_path, video_path):
    """
    Loads the YOLO model and opens the video file.

    :param model_path: Path to the trained YOLO model.
    :param video_path: Path to the input video file.
    :return: Tuple containing the YOLO model and a cv2.VideoCapture object.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    return model, cap


def extract_detections(frame, model, conf_thresh, frame_idx):
    """
    Runs object detection on a frame and extracts detections.

    :param frame: Input image frame.
    :param model: Trained YOLO model.
    :param conf_thresh: Minimum confidence threshold.
    :param frame_idx: Index of the current frame.
    :return: List of detection dictionaries with position and confidence info.
    """
    results = model.predict(frame, imgsz=640, conf=conf_thresh)
    detections = results[0].boxes

    dets = []
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0].item()
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)
        dets.append(
            {
                "frame": frame_idx,
                "cx": cx,
                "cy": cy,
                "confidence": conf,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    return dets


def compute_distances(dets, prev_cx, prev_cy):
    """
    Computes Euclidean distances between detections and previous position.

    :param dets: List of current detections.
    :param prev_cx: Previous x-coordinate.
    :param prev_cy: Previous y-coordinate.
    :return: Updated detections with 'dist' field.
    """
    for d in dets:
        d["dist"] = math.hypot(d["cx"] - prev_cx, d["cy"] - prev_cy)
    return dets


def filter_detections(dets, prev_cx, max_jump_px):
    """
    Filters detections based on confidence and distance threshold.

    :param dets: List of detections with optional 'dist'.
    :param prev_cx: Previous x-coordinate.
    :param max_jump_px: Maximum allowed distance for low-confidence detections.
    :return: Filtered list of detections.
    """
    filtered = []
    for d in dets:
        if d["confidence"] >= 0.5:
            filtered.append(d)
        elif prev_cx is not None and d.get("dist", 0) <= max_jump_px:
            filtered.append(d)
    return filtered


def select_best_detection(dets, prev_cx):
    """
    Selects the best detection based on confidence or distance.

    :param dets: Filtered list of detections.
    :param prev_cx: Previous x-coordinate or None.
    :return: Selected detection.
    """
    if prev_cx is None:
        return max(dets, key=lambda d: d["confidence"])

    for det in dets:
        if "dist" not in det:
            det["dist"] = abs(det["cx"] - prev_cx)

    selected = min(dets, key=lambda d: d["dist"])

    if selected["confidence"] < 0.5:
        high_conf_dets = [d for d in dets if d["confidence"] >= 0.5]
        if high_conf_dets:
            return max(high_conf_dets, key=lambda d: d["confidence"])

    return selected


def process_video_and_get_positions(
    video_path,
    model_path="runs/detect/train25/weights/best.pt",
    conf_thresh=0.05,
):
    """
    Processes a video to track the position of a detected object (e.g., a ball)
    using a YOLO model. Applies filtering to handle occlusions and false positives.

    :param video_path: Path to the input video file.
    :param model_path: Path to the YOLO model weights.
    :param conf_thresh: Minimum detection confidence threshold.
    :return: Tuple containing:
        - records: List of best detections per frame.
        - multiple_balls: List of frame indices with multiple detections.
    """
    model_path = pathlib.Path(model_path)
    video_path = pathlib.Path(video_path)

    model, cap = load_model_and_video(model_path, video_path)

    records, multiple_balls = [], []
    frame_idx = 0
    prev_cx, prev_cy = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dets = extract_detections(frame, model, conf_thresh, frame_idx)

        if len(dets) > 1:
            multiple_balls.append(frame_idx)

        if not dets:
            frame_idx += 1
            continue

        if prev_cx is not None:
            dets = compute_distances(dets, prev_cx, prev_cy)

        if len(dets) == 1:
            d = dets[0]
            if (
                d["confidence"] < 0.5
                and prev_cx is not None
                and d["dist"] > MAX_PIXEL_JUMP
            ):
                frame_idx += 1
                continue

        dets = filter_detections(dets, prev_cx, MAX_PIXEL_JUMP)

        if not dets:
            frame_idx += 1
            continue

        best = select_best_detection(dets, prev_cx)
        records.append(best)
        prev_cx, prev_cy = best["cx"], best["cy"]
        frame_idx += 1

    cap.release()
    return records, multiple_balls
