"""
This module extracts data from videos using Tesseract OCR.
It reads player names, set scores, and points, and identifies match start times
to segment the video into individual matches for further analysis.
"""

from pathlib import Path

import cv2
import pandas as pd
from pytesseract import pytesseract
from skimage.metrics import structural_similarity as ssim  # pylint: disable=no-name-in-module
from src.tt_track import constants as c


def read_name(frame, top_left, bottom_right):
    """
    Extract player names from a specified region in the video frame using OCR.

    :param frame: The input video frame (image).
    :param top_left: Tuple (x1, y1) representing the top-left corner of the region to crop.
    :param bottom_right: Tuple (x2, y2) representing the bottom-right corner of the region to crop.
    :return: List of tuples (player_index, player_name) for two players.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    names = []
    for player_idx in (0, 1):
        dy = player_idx * c.PLAYER_SHIFT_Y
        crop = frame[y1 + dy: y2 + dy, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(crop, crop, mask=thr)
        text = pytesseract.image_to_string(
            cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), config=c.PLAYER_NAME_CONFIG
        ).strip()
        names.append((player_idx + 1, text))
    return names


# This function reads player names with a vertical shift between players.
# It differs from my initial approach, which read all points line-by-line.
# Later, I switched to reading just the points from the last set,
# so this function has a different structure compared to others.
def read_set(frame, player_shift, top_left, bottom_right):
    """

    :param player_shift:
    :param frame:
    :param top_left:
    :param bottom_right:
    :return:
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    dy = player_shift * c.PLAYER_SHIFT_Y
    crop = frame[y1 + dy: y2 + dy, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(crop, crop, mask=thr)
    text = pytesseract.image_to_string(
        cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), config=c.CUSTOM_CONFIG
    ).strip()
    return text

# Used for creating a dataset for OCR to later develop a custom Tesseract font
# to improve accuracy. Currently, the standard Tesseract works fine,
# but using a custom font could be good for future upgrades.

# def save_crop_image(crop, save_dir, crop_type, frame_idx, player_idx):
#     """
#
#     :param crop:
#     :param save_dir:
#     :param crop_type:
#     :param frame_idx:
#     :param player_idx:
#     :return:
#     """
#     if save_dir:
#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         subdir = Path(save_dir) / crop_type
#         subdir.mkdir(parents=True, exist_ok=True)
#         filename = subdir / f"{crop_type}_f{frame_idx:05d}_p{player_idx}.png"
#         cv2.imwrite(str(filename), crop)


def read_last_points(frame, top_left, bottom_right, last_set_index):
    """
    Extract point values for the last completed set from the score column.

    Args:
        frame (ndarray): The video frame image.
        top_left (tuple): (x, y) coordinates of the top-left corner of the score area.
        bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the score area.
        last_set_index (int): 1-based index of the last completed set (1 for first set, etc.).

    Returns:
        list of tuples: [(1, pts1), (2, pts2)] where pts1 and pts2 are point strings for players 1 and 2.
    """
    x1 = top_left[0] + c.SCORE_SHIFT_X * last_set_index
    x2 = bottom_right[0] + c.SCORE_SHIFT_X * last_set_index

    col = []
    for player_idx in (0, 1):
        dy = player_idx * c.PLAYER_SHIFT_Y
        crop = frame[top_left[1] + dy: bottom_right[1] + dy, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(crop, crop, mask=thr)
        pts = pytesseract.image_to_string(
            cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), config=c.CUSTOM_CONFIG
        ).strip()

        col.append((player_idx + 1, pts))

    return col

# This function is simple, as I initially had more heuristics to filter out results,
# but later found that filtering only when both players' data are complete works better.
def check_pair(col):
    """
    Check if both players have non-empty OCR results.

    Args:
        col (list of tuples): List containing tuples of (player_index, points_str).

    Returns:
        bool: True if both players' points strings are non-empty, False otherwise.
    """
    return all(pts != "" for _, pts in col)


# I tried also setting the cap, but it is faster to just go through all frames
def process_video_data(video_path):
    """
    Processes a video file to extract player names, sets, and points using OCR.

    :param video_path: Path to the video file.
    :type video_path: str
    :return: Dictionary with player names and frame-wise sets and points.
    :rtype: dict
    :raises RuntimeError: If the video cannot be opened or a frame cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame at {frame_count // 2}")

    player_names = read_name(frame, c.LEFT_TOP_COR_PLAYER, c.RIGHT_BOTTOM_COR_PLAYER)

    results = {"players": dict(player_names), "frames": []}
    cap = cv2.VideoCapture(video_path)
    previous_points = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % c.FRAME_INTERVAL != 0:
            continue

        sets = []
        total_sets = 0
        for p in (0, 1):
            st = read_set(frame, p, c.LEFT_TOP_COR_SET, c.RIGHT_BOTTOM_COR_SET)
            sets.append(st)
            try:
                total_sets += int(st)
            except ValueError:
                total_sets = -1
                break

        if total_sets < 0 or total_sets > 5 or "" in sets:
            continue

        last_points = read_last_points(
            frame, c.LEFT_TOP_COR_SCORE, c.RIGHT_BOTTOM_COR_SCORE, total_sets
        )

        if not check_pair(last_points):
            if previous_points is None:
                continue
            last_points = previous_points

        if last_points == previous_points:
            continue

        if (
                int(last_points[0][1]) + int(last_points[1][1]) > 21
                and abs(int(last_points[0][1]) - int(last_points[1][1])) >= 2
        ):
            continue

        frame_data = {
            "frame": frame_idx,
            "sets": {1: sets[0], 2: sets[1]},
            "points": [
                {"set": total_sets, "player": p, "points": pts}
                for p, pts in last_points
            ],
        }
        results["frames"].append(frame_data)
        previous_points = last_points

    cap.release()
    return results


# @generated "[ALL]" ChatGPT o4: [Prompt: Create a function to take latest file in dict]
def get_latest_file(directory):
    """
    Returns the most recently created file in the given directory.

    :param directory: Path to the directory containing the video files.
    :return: The path of the latest file or None if the directory is empty.
    :rtype: pathlib.Path or None
    """
    files = list(Path(directory).glob("*"))
    files = [f for f in files if f.is_file()]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_ctime)

# This logic was originally inside the match segments function,
# but to comply with PEP8’s recommendation on limiting local variables,
# I moved it to a separate function.
def build_match_segments(blocks, names, fps):
    """
    Builds match segments with timing and player information based on frame blocks.

    :param blocks: List of tuples or lists representing frame ranges (start_frame, end_frame).
    :type blocks: list of tuples/lists with two integers
    :param names: List of dictionaries containing player names for each segment.
    :type names: list of dict with keys 'player1_name' and 'player2_name'
    :param fps: Frames per second of the video.
    :type fps: float or int
    :return: List of dictionaries, each describing a match segment with frame indices, times, duration, and player names.
    :rtype: list of dict
    """
    segments = []
    for i in range(len(blocks) - 1):
        start_frame = blocks[i][1]
        end_frame = blocks[i + 1][0]
        segments.append(
            {
                "match_start_frame": start_frame,
                "match_end_frame": end_frame,
                "match_start_time_sec": start_frame / fps,
                "match_end_time_sec": end_frame / fps,
                "duration_sec": (end_frame - start_frame) / fps,
                "player1_name": names[i]["player1_name"],
                "player2_name": names[i]["player2_name"],
            }
        )
    return segments


def extract_match_segments(
        video_path: str,
        background_image_path: str,
        interval_sec: int = c.FRAME_INTERVAL_SECONDS,
) -> pd.DataFrame:
    """
    Extracts match segments from a video by comparing frames against a background image.

    Detects blocks of frames where the content differs significantly from the background,
    indicating match activity, and captures player names for each match segment.

    :param video_path: Path to the video file.
    :param background_image_path: Path to the background image used for frame comparison.
    :param interval_sec: Time interval in seconds between frames to sample for analysis.
                         Defaults to c.FRAME_INTERVAL_SECONDS.
    :return: A pandas DataFrame containing the match segments with frame indices, times,
             and player names for each segment.
    """
    background_image_path = Path(background_image_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    video_info = {
        "back_gray": cv2.cvtColor(
            cv2.imread(background_image_path), cv2.COLOR_BGR2GRAY
        ),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    video_info["frame_interval"] = int(video_info["fps"] * interval_sec)

    in_block = False
    block_start = None
    blocks = []
    pending_read = False
    names = []

    for frame_idx in range(0, video_info["frame_count"], video_info["frame_interval"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if video_info["back_gray"].shape != frame_gray.shape:
            frame_gray = cv2.resize(
                frame_gray,
                (video_info["back_gray"].shape[1], video_info["back_gray"].shape[0]),
            )

        if pending_read:
            name = read_name(frame, c.LEFT_TOP_COR_PLAYER, c.RIGHT_BOTTOM_COR_PLAYER)
            names.append(
                {
                    "start_frame": blocks[-1][0],
                    "end_frame": blocks[-1][1],
                    "start_time_sec": blocks[-1][0] / video_info["fps"],
                    "end_time_sec": blocks[-1][1] / video_info["fps"],
                    "player1_name": name[0][1],
                    "player2_name": name[1][1],
                }
            )
            pending_read = False
        # Using SSIM to handle occasional video artifacts introduced by YouTube, which makes perfect match unusable
        if ssim(video_info["back_gray"], frame_gray) > c.THRESHOLD:
            if not in_block:
                block_start = frame_idx
                in_block = True
        else:
            if in_block:
                blocks.append((block_start, frame_idx - video_info["frame_interval"]))
                in_block = False
                pending_read = True

    if in_block:
        blocks.append((block_start, frame_idx))
        name = read_name(frame, c.LEFT_TOP_COR_PLAYER, c.RIGHT_BOTTOM_COR_PLAYER)
        names.append(
            {
                "start_frame": block_start,
                "end_frame": frame_idx,
                "start_time_sec": block_start / video_info["fps"],
                "end_time_sec": frame_idx / video_info["fps"],
                "player1_name": name[0][1],
                "player2_name": name[1][1],
            }
        )

    cap.release()

    return build_match_segments(blocks, names, video_info["fps"])


def prepare_numeric_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert point columns in the DataFrame to numeric values.

    Invalid or non-numeric strings are coerced to NaN, then replaced with -1,
    and finally converted to integers.

    :param df: Input DataFrame containing 'player_1_points' and 'player_2_points' columns.
    :return: A new DataFrame with 'player_1_points' and 'player_2_points' columns
             converted to integers, with invalid entries replaced by -1.
    """
    df = df.copy()
    df["player_1_points"] = (
        pd.to_numeric(df["player_1_points"], errors="coerce").fillna(-1).astype(int)
    )
    df["player_2_points"] = (
        pd.to_numeric(df["player_2_points"], errors="coerce").fillna(-1).astype(int)
    )
    return df


def filter_outliers_by_sliding_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outlier rows based on sliding windows of size 3.

    For each window of three consecutive rows, the middle row is removed if,
    for both players, the middle point is not between its immediate neighbors' points,
    considering only rows within the same set.

    :param df: Input DataFrame with columns:
               'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points'.
    :return: Filtered DataFrame with outlier rows removed.
    """
    df = df.copy()

    mask = [True] * len(df)

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        mid = df.iloc[i]
        nxt = df.iloc[i + 1]

        if (
                (prev["player_1_sets"], prev["player_2_sets"])
                == (mid["player_1_sets"], mid["player_2_sets"])
                == (nxt["player_1_sets"], nxt["player_2_sets"])
        ):
            if not (
                    min(prev["player_1_points"], nxt["player_1_points"])
                    <= mid["player_1_points"]
                    <= max(prev["player_1_points"], nxt["player_1_points"])
            ):
                mask[i] = False
                continue
            if not (
                    min(prev["player_2_points"], nxt["player_2_points"])
                    <= mid["player_2_points"]
                    <= max(prev["player_2_points"], nxt["player_2_points"])
            ):
                mask[i] = False
                continue

    return df[mask].reset_index(drop=True)


def filter_backward_point_jumps(df: pd.DataFrame, tolerance: int = 4) -> pd.DataFrame:
    """
    Remove rows where either player's points decrease significantly within the same set.

    A small tolerance is allowed for point decreases (default is 2).

    :param df: Input DataFrame containing players' sets and points columns.
    :param tolerance: Maximum allowed decrease in points before a row is removed.
    :return: Filtered DataFrame with rows exhibiting large backward jumps removed.
    """
    df = df.copy()

    mask = [True] * len(df)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if (prev["player_1_sets"], prev["player_2_sets"]) == (
                curr["player_1_sets"],
                curr["player_2_sets"],
        ):
            drop_1 = prev["player_1_points"] - curr["player_1_points"]
            drop_2 = prev["player_2_points"] - curr["player_2_points"]

            if drop_1 > tolerance or drop_2 > tolerance:
                mask[i] = False

    return df[mask].reset_index(drop=True)


def fill_starting_zero_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each new set starts with 0:0 points by inserting a zero-points row
    if the set's first row does not have points (0, 0).

    :param df: DataFrame with columns 'player_1_points', 'player_2_points',
               'player_1_sets', 'player_2_sets', and 'frame'.
    :return: DataFrame with zero-points rows added at the start of each set,
             sorted by 'frame' and reindexed.
    """
    df = df.copy()
    result = []
    prev_set = None

    for _, row in df.iterrows():
        current_set = (row["player_1_sets"], row["player_2_sets"])

        if current_set != prev_set:
            if (row["player_1_points"], row["player_2_points"]) != (0, 0):
                zero_row = row.copy()
                zero_row["player_1_points"] = 0
                zero_row["player_2_points"] = 0
                zero_row["frame"] = row["frame"] - 1
                result.append(zero_row)

            prev_set = current_set

        result.append(row)

    return pd.DataFrame(result).sort_values(by="frame").reset_index(drop=True)


def fill_missing_points_inside_sets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates missing intermediate points within each set by inserting rows
    between consecutive frames. Points increments for player 1 are applied first,
    followed by player 2’s increments.

    :param df: DataFrame containing columns 'player_1_points', 'player_2_points',
               'player_1_sets', 'player_2_sets', and 'frame'.
    :return: DataFrame with interpolated rows filling missing points inside sets,
             sorted by 'frame' and reindexed.
    """
    df = df.copy()
    result = [df.iloc[0].copy()]

    for i in range(len(df) - 1):
        prev = df.iloc[i]
        nxt = df.iloc[i + 1]

        if (prev.player_1_sets, prev.player_2_sets) == (
                nxt.player_1_sets,
                nxt.player_2_sets,
        ):
            dp1 = nxt.player_1_points - prev.player_1_points
            dp2 = nxt.player_2_points - prev.player_2_points
            total_steps = abs(dp1) + abs(dp2)

            if total_steps > 0:
                for s in range(1, total_steps):

                    cur1 = prev.player_1_points + (
                            min(s, abs(dp1)) * (1 if dp1 > 0 else -1)
                    )
                    advance2 = max(0, s - abs(dp1))
                    cur2 = prev.player_2_points + (advance2 * (1 if dp2 > 0 else -1))

                    frac = s / total_steps
                    frame = int(prev.frame + (nxt.frame - prev.frame) * frac)

                    new_row = prev.copy()
                    new_row["player_1_points"] = cur1
                    new_row["player_2_points"] = cur2
                    new_row["frame"] = frame
                    result.append(new_row)

        result.append(nxt.copy())

    result = pd.DataFrame(result)
    result = result[df.columns]
    return result.sort_values("frame").reset_index(drop=True).astype(int)

# This function is a bit unecessory, but I think it makes the final code more readable
def is_set_finished(p1, p2):
    """
    Determine if a set is finished based on players' points.

    :param p1: Points scored by player 1 (int).
    :param p2: Points scored by player 2 (int).
    :return: True if the set is finished (a player has at least 11 points and
             leads by at least 2 points), False otherwise.
    """
    return (p1 >= 11 or p2 >= 11) and abs(p1 - p2) >= 2


def winner_player_from_sets(set_curr, set_next):
    """
    Determine which player won the next set based on set scores.

    :param set_curr: Tuple[int, int] - current set scores (player1, player2).
    :param set_next: Optional[Tuple[int, int]] - next set scores (player1, player2) or None.
    :return: int or None - 1 if player 1 won the next set, 2 if player 2 won,
             None if no winner or next set is None.
    """
    if set_next is None:
        return None
    p1_curr, p2_curr = set_curr
    p1_next, p2_next = set_next

    if p1_next > p1_curr:
        return 1
    if p2_next > p2_curr:
        return 2
    return None


def complete_sets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every set is completed according to table tennis rules.

    For each set in the DataFrame, if the last recorded score does not
    meet the winning condition (at least 11 points and a 2-point difference),
    this function appends rows incrementing points until the set is finished.

    The winner of a set is inferred by checking which player has increased
    their set count in the subsequent set. If no such information is available,
    the player currently leading or player 1 in case of a tie is considered
    the winner.

    :param df: Input DataFrame with columns 'player_1_points', 'player_2_points',
               'player_1_sets', 'player_2_sets', and 'frame'.
    :return: DataFrame with completed sets, including artificially added rows
             to represent final scoring points of unfinished sets.
    """

    df = df.copy()
    result = []
    current_set = None
    last_row = {}

    for _, row in df.iterrows():
        set_id = (row["player_1_sets"], row["player_2_sets"])

        if current_set is not None and set_id != current_set:
            p1 = last_row["player_1_points"]
            p2 = last_row["player_2_points"]
            frame = last_row["frame"]
            set1 = last_row["player_1_sets"]
            set2 = last_row["player_2_sets"]

            winner = winner_player_from_sets(current_set, set_id)

            while not is_set_finished(p1, p2):
                if winner == 1:
                    p1 += 1
                elif winner == 2:
                    p2 += 1
                else:
                    if p1 >= p2:
                        p1 += 1
                    else:
                        p2 += 1

                frame += 1
                result.append(
                    {
                        "player_1_sets": set1,
                        "player_2_sets": set2,
                        "player_1_points": p1,
                        "player_2_points": p2,
                        "frame": frame,
                    }
                )

        result.append(row.to_dict())
        current_set = set_id
        last_row = row

    if last_row is not None:
        p1, p2 = last_row["player_1_points"], last_row["player_2_points"]
        frame = last_row["frame"]
        set1, set2 = last_row["player_1_sets"], last_row["player_2_sets"]

        while not is_set_finished(p1, p2):
            if p1 >= p2:
                p1 += 1
            else:
                p2 += 1

            frame += 1
            result.append(
                {
                    "player_1_sets": set1,
                    "player_2_sets": set2,
                    "player_1_points": p1,
                    "player_2_points": p2,
                    "frame": frame,
                }
            )

    return pd.DataFrame(result).sort_values("frame").reset_index(drop=True).astype(int)


def remove_duplicate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove consecutive duplicate rows where both players' points and set counts are unchanged.

    Consecutive rows with identical values in 'player_1_points', 'player_2_points',
    'player_1_sets', and 'player_2_sets' are considered duplicates, and only the first
    occurrence is retained.

    :param df: Input DataFrame containing player points and set counts.
    :return: DataFrame with consecutive duplicate score rows removed and index reset.
    """
    df = df.copy()

    state_cols = [
        "player_1_points",
        "player_2_points",
        "player_1_sets",
        "player_2_sets",
    ]


    is_duplicate = (df[state_cols] == df[state_cols].shift()).all(axis=1)

    return df[~is_duplicate].reset_index(drop=True)


def correcting_points_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a sequence of corrections to the dataset to ensure point data consistency.

    The processing steps include:
    1. Convert points to numeric and handle invalid values.
    2. Remove outlier rows based on sliding window filtering.
    3. Filter out backward jumps in points within sets.
    4. Remove consecutive duplicate score rows.
    5. Ensure each set starts at 0:0 points.
    6. Fill missing intermediate points within sets by interpolation.
    7. Complete unfinished sets by adding points until a winner is determined.

    :param df: Input DataFrame with raw point and set data.
    :return: DataFrame with cleaned and corrected point data.
    """
    df_num = prepare_numeric_points(df)
    df_no_outliers = filter_outliers_by_sliding_window(df_num)
    df_no_jumps = filter_backward_point_jumps(df_no_outliers)
    df_no_duplicates = remove_duplicate_scores(df_no_jumps)
    df_filled_start = fill_starting_zero_points(df_no_duplicates)
    df_filled_all = fill_missing_points_inside_sets(df_filled_start)
    df_final = complete_sets(df_filled_all)
    return df_final


def results_to_summary_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert processed video data results into a structured pandas DataFrame.

    The DataFrame contains one row per frame with the following columns:
        - frame: Frame index
        - player_1_sets: Number of sets won by player 1 (int)
        - player_2_sets: Number of sets won by player 2 (int)
        - player_1_points: OCR-detected points for player 1 in the current set (int or empty string)
        - player_2_points: OCR-detected points for player 2 in the current set (int or empty string)

    :param results: Dictionary output from process_video_data containing frame-wise sets and points data.
    :return: pandas DataFrame summarizing the match data by frame.
    """
    rows = []
    for frame_info in results["frames"]:
        frame = frame_info["frame"]
        sets_dict = frame_info["sets"]
        points = {p["player"]: p["points"] for p in frame_info["points"]}
        rows.append(
            {
                "frame": frame,
                "player_1_sets": int(sets_dict[1]),
                "player_2_sets": int(sets_dict[2]),
                "player_1_points": points.get(1, ""),
                "player_2_points": points.get(2, ""),
            }
        )
    return pd.DataFrame(rows)


def add_rally_times(df, frame_rate=30):
    """
    Calculate and add rally times (duration between consecutive frames) to the DataFrame.

    Rally time is computed as the difference between consecutive 'frame' values divided by the frame rate.
    The last row will have NaN for rally time since it has no following frame.

    :param df: pandas DataFrame containing a 'frame' column with frame indices (int).
    :param frame_rate: Frame rate (frames per second) used to convert frame differences into seconds. Default is 30.
    :return: A copy of the input DataFrame with an added 'rallie_time' column of type float (seconds).
    """
    rallie_time = []
    for i in range(len(df) - 1):
        prev_frame = df.iloc[i]["frame"]
        next_frame = df.iloc[i + 1]["frame"]
        rallie_time.append((next_frame - prev_frame) / frame_rate)
    rallie_time.append(pd.NA)  # Last row has no next rally
    df = df.copy()
    df["rallie_time"] = pd.Series(rallie_time, dtype="Float64")
    return df

# Always the same player starts serving
def deduce_server(total_points, first_server):
    """
    Determines the current server based on total points played and the first server.

    :param total_points: Total number of points played in the game.
    :param first_server: Player number who served first (typically 1 or 2).
    :return: Player number who is currently serving.
    """
    if total_points < 20:
        if (total_points // 2) % 2 == 0:
            return first_server
        return 3 - first_server

    if total_points % 2 == 0:
        return first_server
    return 3 - first_server


def add_servers(df):
    """
    Add a 'server' column indicating which player is serving each point.

    The first server in each set alternates based on the set number (sum of sets won).
    The server is deduced from the total points played in the current set and the first server.

    :param df: pandas DataFrame with columns 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points'.
    :return: A copy of the DataFrame with an added 'server' column (int, 1 or 2).
    """
    servers = []
    for _, row in df.iterrows():
        player_1_sets = row["player_1_sets"]
        player_2_sets = row["player_2_sets"]
        total_points = row["player_1_points"] + row["player_2_points"]
        set_number = player_1_sets + player_2_sets
        first_server = 1 if (set_number % 2 == 0) else 2
        servers.append(deduce_server(total_points, first_server))

    df = df.copy()
    df["server"] = servers
    return df


def add_winners(df):
    """
    Add a 'won' column indicating the player who won each point.

    Compares consecutive rows' points to determine which player scored the point.
    The last row is assigned pd.NA since no subsequent point exists.

    :param df: pandas DataFrame with columns 'player_1_points', 'player_2_points'.
    :return: A copy of the DataFrame with an added 'won' column (Int64), 1 or 2 indicating player or NA.
    """
    won = []
    for i in range(len(df) - 1):
        prev = df.iloc[i]
        nxt = df.iloc[i + 1]
        if nxt["player_1_points"] > prev["player_1_points"]:
            won.append(1)
        elif nxt["player_2_points"] > prev["player_2_points"]:
            won.append(2)
        else:
            won.append(pd.NA)
    # Padding for last row, it has no winner
    won.append(pd.NA)
    df = df.copy()
    df["won"] = pd.Series(won, dtype="Int64")
    return df


def add_match_info(df):
    """
    Enrich the DataFrame by adding server, winner, and rally time information.

    Sequentially applies:
      - add_servers
      - add_winners
      - add_rally_times

    :param df: pandas DataFrame containing match point data.
    :return: DataFrame with added 'server', 'won', and 'rallie_time' columns.
    """
    df = add_servers(df)
    df = add_winners(df)
    df = add_rally_times(df)
    return df
