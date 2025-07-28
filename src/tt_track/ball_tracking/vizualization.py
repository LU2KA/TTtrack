"""
This module implements visualization of tracked balls and also bounces visualization.
"""

from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tt_track.constants import TABLE_POLYGON

# function I used during testing, to dislay video
# def display_ball_positions(video_path, records):
#     """
#
#     :param video_path:
#     :param records:
#     :return:
#     """
#     cap = cv2.VideoCapture(video_path)
#     frame_idx = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         detections_in_frame = [rec for rec in records if rec["frame"] == frame_idx]
#
#         for det in detections_in_frame:
#             cx, cy = int(det["cx"]), int(det["cy"])
#             x1, y1, x2, y2 = (
#                 int(det["x1"]),
#                 int(det["y1"]),
#                 int(det["x2"]),
#                 int(det["y2"]),
#             )
#
#             cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#             conf_text = f"{det['confidence']:.2f}"
#             cv2.putText(
#                 frame,
#                 conf_text,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 255),
#                 2,
#             )
#
#         cv2.imshow("Ball Detection", frame)
#         cv2.waitKey(50)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#         frame_idx += 1
#     cap.release()
#     cv2.destroyAllWindows()


def draw_detections(frame, detections, frame_idx, trail):
    """
    Draws ball detections on the given frame and updates the trail.

    :param frame: The video frame to draw the detections on.
    :param detections: A list of detected ball positions and metadata.
    :param frame_idx: The index of the current frame.
    :param trail: A list storing previous ball positions for drawing the trail.
    :return: None
    """

    for det in detections:
        # Also being paranoid here — only cx, cy should be floats,
        # but I got a float error, so I wrapped everything just in case.
        cx, cy = int(det["cx"]), int(det["cy"])
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{det['confidence']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        trail.append((frame_idx, (cx, cy)))


def draw_trail(frame, trail):
    """
    Draw the trail of ball positions.

    :param frame: Frame image to draw on.
    :param trail: List of tuples (frame_idx, (x, y)) representing ball positions.
    :return: None
    """
    for (_, pt1), (_, pt2) in zip(trail, trail[1:]):
        cv2.line(frame, pt1, pt2, (0, 200, 200), 2)


def draw_collisions(frame, frame_idx, collision_frames, collision_records, highlight_frames):
    """
    Highlight collision points if within highlight range.

    :param frame: Frame image to draw on.
    :param frame_idx: Current frame index.
    :param collision_frames: List of frames where collisions occurred.
    :param collision_records: List of collision data dicts.
    :param highlight_frames: Number of frames to highlight collisions.
    :return: None
    """
    for coll_frame in collision_frames:
        if coll_frame <= frame_idx <= coll_frame + highlight_frames:
            for r in collision_records:
                if r["frame"] == coll_frame:
                    cx = int(r["cx"])
                    cy = int(r["cy"])
                    cv2.circle(frame, (cx, cy), 1, (0, 0, 255), 3)
                    break


def display_ball_trajectory_with_collisions_mp4(
    video_path,
    records,
    collision_records,
    highlight_frames=30,
    output_path="./output_trajectory.mp4",
):
    """
    Save ball positions, trajectory, and highlight collisions to a video file.
    Overlays can include bounding boxes, trails, and collision markers.

    :param video_path: Path to the input video file.
    :param records: List of ball detection records.
    :param collision_records: List of collision detection records.
    :param highlight_frames: Number of frames to highlight collisions (default 30).
    :param output_path: Path to save the output video file.
    :return: Path to the saved output video.
    """
    output_path = Path(output_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    trail = []
    rec_idx = 0
    collision_frames = [rec["frame"] for rec in collision_records]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        Path(output_path),
        fourcc,
        int(cap.get(cv2.CAP_PROP_FPS)),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # the 10 is the decay time for trajectory
        trail = [(f, pt) for (f, pt) in trail if f > frame_idx - 10]

        frame_detections = []
        while rec_idx < len(records) and records[rec_idx]["frame"] == frame_idx:
            frame_detections.append(records[rec_idx])
            rec_idx += 1

        draw_detections(frame, frame_detections, frame_idx, trail)
        draw_trail(frame, trail)
        draw_collisions(
            frame, frame_idx, collision_frames, collision_records, highlight_frames
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_path}")
    return output_path


def map_and_plot_bounces_topdown(
    collisions,
    table_polygon=TABLE_POLYGON,
    output_size=(548, 305),
    table_image_path=None,
):
    """
    Create a top-down view plot of collision points mapped onto a table polygon.

    :param collisions: List of collision dicts with 'cx', 'cy' coordinates.
    :param table_polygon: List of polygon points defining the table.
    :param output_size: Output image size (width, height).
    :param table_image_path: Optional path to table background image.
    :return: Matplotlib figure object.
    """
    width, height = output_size
    dst = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )

    src = np.array(table_polygon, dtype=np.float32)

    # This function uses Homography instead of a simple perspective transform.
    # Initially, I tried using perspective transform with more points to make it work,
    # but later discovered that the table had moved, so I switched to the simpler approach.
    m_matrix, _ = cv2.findHomography(src, dst)

    pts = np.array([[c["cx"], c["cy"]] for c in collisions], dtype=np.float32).reshape(-1, 1, 2)  # pylint: disable=too-many-function-args
    warped_pts = cv2.perspectiveTransform(pts, m_matrix).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(3)

    ax.set_title("Top-Down View of Bounces", fontsize=16)

    if table_image_path:
        img = mpimg.imread(table_image_path)
        ax.imshow(img, extent=[0, width, height, 0])

    ax.scatter(
        warped_pts[:, 0],
        warped_pts[:, 1],
        c="white",
        edgecolors="black",
        s=80,
        label="Bounces",
        zorder=3,
    )

    ax.legend(loc="upper right", fontsize=10)

    return fig


def map_and_plot_regional_heatmap(
    collisions, table_polygon=TABLE_POLYGON, output_size=(600, 300), grid_size=(10, 6)
):
    """
    Create a regional heatmap of collisions mapped onto the table polygon.

    :param collisions: List of collision dicts with 'cx', 'cy' coordinates.
    :param table_polygon: List of polygon points defining the table.
    :param output_size: Output image size (width, height).
    :param grid_size: Grid size for heatmap bins (cols, rows).
    :return: Matplotlib figure object.
    """

    width, height = output_size

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    m_matrix = cv2.getPerspectiveTransform(np.array(table_polygon, dtype=np.float32), dst)

    pts = np.array([[c["cx"], c["cy"]] for c in collisions], dtype=np.float32)
    warped_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), m_matrix).reshape(-1, 2)  # pylint: disable=too-many-function-args

    # This just splits the table into the grid size and then match the corresponding part
    x_bins = np.linspace(0, width, grid_size[0] + 1)
    y_bins = np.linspace(0, height, grid_size[1] + 1)
    heatmap, _, _ = np.histogram2d(
        warped_pts[:, 0], warped_pts[:, 1], bins=[x_bins, y_bins]
    )
    heatmap = heatmap.T

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        pd.DataFrame(heatmap),
        cmap="rocket_r",
        alpha=0.85,
        cbar_kws={"label": "Bounce Count"},
        ax=ax,
        annot=True,
        xticklabels=False,
        yticklabels=False,
        square=False,
        cbar=False,
    )

    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(grid_size[1], 0)
    ax.set_title("Regional Bounce Heatmap", fontsize=14, pad=20)

    sns.despine(left=True, bottom=True)

    return fig
