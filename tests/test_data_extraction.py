import time
from pathlib import Path

import pandas as pd
import pytest
from src.tt_track import constants as c
from src.tt_track.data_extraction.extract_match_data import read_name, read_set, read_last_points,check_pair, process_video_data, get_latest_file, extract_match_segments, results_to_summary_dataframe,process_video_data, filter_outliers_by_sliding_window, filter_backward_point_jumps,fill_starting_zero_points,fill_missing_points_inside_sets,is_set_finished,complete_sets, remove_duplicate_scores,deduce_server,add_match_info
import cv2

@pytest.mark.parametrize("filename, expected_names", [
    ("frame_test0.jpg", [(1, 'JICHA D.'), (2, 'STREJC F.')]),
    ("frame_test1.jpg", [(1, 'KOLAR V.'), (2, 'KOTRBATY P.')]),
    ("frame_test2.jpg", [(1, 'VORISEK T.'), (2, 'SILHAN P.')]),
    ("frame_test_background.jpg", [(1, ''), (2, '')]),
])
def test_read_name_from_image(filename, expected_names):
    frame_path = Path(__file__).parent / "data" / filename
    frame = cv2.imread(frame_path)
    assert frame is not None, f"Image {filename} not loaded correctly"

    names = read_name(frame, c.LEFT_TOP_COR_PLAYER, c.RIGHT_BOTTOM_COR_PLAYER)
    assert names == expected_names

@pytest.mark.parametrize("filename, player_1_sets, player_2_sets", [
    ("frame_test0.jpg", '2', '0'),
    ("frame_test1.jpg", '2', '2'),
    ("frame_test2.jpg", '1', '0'),
    ("frame_test_background.jpg", '', ''),
])
def test_read_set_from_image(filename, player_1_sets, player_2_sets):
    frame_path = Path(__file__).parent / "data" / filename
    frame = cv2.imread(frame_path)
    assert frame is not None, f"Image {filename} not loaded correctly"

    set0 = read_set(frame, 0, c.LEFT_TOP_COR_SET, c.RIGHT_BOTTOM_COR_SET)
    set1 = read_set(frame, 1, c.LEFT_TOP_COR_SET, c.RIGHT_BOTTOM_COR_SET)
    assert set0 == player_1_sets
    assert set1 == player_2_sets

@pytest.mark.parametrize("filename, last_set_index, expected_points", [
    ("frame_test0.jpg", 2, [(1, '10'), (2, '9')]),
    ("frame_test0.jpg", 0, [(1, '11'), (2, '8')]),
    ("frame_test1.jpg", 2, [(1, '9'), (2, '11')]),
    ("frame_test1.jpg", 3, [(1, '7'), (2, '11')]),
    ("frame_test1.jpg", 4, [(1, '10'), (2, '9')]),
    ("frame_test2.jpg", 1, [(1, '3'), (2, '3')]),
    ("frame_test2.jpg", 0, [(1, '11'), (2, '8')]),
    ("frame_test_background.jpg", 0, [(1, ''), (2, '')]),
])
def test_read_last_points_from_image(filename, last_set_index, expected_points):
    frame_path = Path(__file__).parent / "data" / filename
    frame = cv2.imread(str(frame_path))
    assert frame is not None, f"Image {filename} not loaded correctly"

    points = read_last_points(frame, c.LEFT_TOP_COR_SCORE, c.RIGHT_BOTTOM_COR_SCORE, last_set_index)
    assert points == expected_points

@pytest.mark.parametrize("col, expected", [
    ([(1, "5"), (2, "3")], True),
    ([(1, "0"), (2, "1")], True),
    ([(1, ""), (2, "1")], False),
    ([(1, "2"), (2, "")], False),
    ([(1, ""), (2, "")], False),
])
def test_check_pair(col, expected):
    assert check_pair(col) == expected

def test_process_video():
    video_path = Path(__file__).parent / "data" / "test.mp4"
    result = process_video_data(video_path)

    assert isinstance(result, dict)
    assert "players" in result
    assert "frames" in result
    assert isinstance(result["players"], dict)
    assert isinstance(result["frames"], list)

    if result["frames"]:
        frame_data = result["frames"][0]
        assert "frame" in frame_data
        assert "sets" in frame_data
        assert "points" in frame_data
        assert isinstance(frame_data["points"], list)

    video_path = Path(__file__).parent / "data" / "nonexistent.mp4"
    with pytest.raises(RuntimeError, match="Cannot open"):
        process_video_data(video_path)

def test_get_latest_file(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("first")
    time.sleep(0.1)

    file2 = tmp_path / "file2.txt"
    file2.write_text("second")
    time.sleep(0.1)

    file3 = tmp_path / "file3.txt"
    file3.write_text("third")

    latest = get_latest_file(tmp_path)
    assert latest == file3
    assert latest.name == "file3.txt"

    file3.unlink()

    latest = get_latest_file(tmp_path)
    assert latest == file2
    assert latest.name == "file2.txt"

    file1.unlink()
    file2.unlink()

    latest = get_latest_file(tmp_path)
    assert latest is None

def test_extract_match_segments():
    video_path = Path(__file__).parent / "data" / "matches_uncut.mp4"
    back = Path(__file__).parent / "data" / "frame_test_background.jpg"
    res = extract_match_segments(video_path,back)
    assert len(res) == 1
    assert 150 <= res[0]['match_start_time_sec'] <= 180
    assert 1842 <= res[0]['match_end_time_sec'] <= 1872
    assert 1602 <= res[0]['duration_sec'] <= 1720
    assert res[0]['player1_name'] == "LANGER A."
    assert res[0]['player2_name'] == "VONASEK J."

def test_sliding_window():
    data1 = [
        [896, 0, 0, 1, 0],
        [1152, 0, 0, 2, 0],
        [1600, 0, 0, 3, 0],
        [2000, 0, 0, 11, 2],
        [2112, 0, 0, 3, 1],
        [2304, 0, 0, 3, 7],
        [2432, 0, 0, 3, 2],
        [2816, 0, 0, 4, 2],
        [3392, 0, 0, 4, 3],
        [4096, 0, 0, 6, 3]
    ]
    res1 = [
        [896, 0, 0, 1, 0],
        [1152, 0, 0, 2, 0],
        [1600, 0, 0, 3, 0],
        [2432, 0, 0, 3, 2],
        [2816, 0, 0, 4, 2],
        [3392, 0, 0, 4, 3],
        [4096, 0, 0, 6, 3]
    ]
    columns = ['frame', 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points']

    df = pd.DataFrame(data1, columns=columns)
    df_res1 = pd.DataFrame(res1, columns=columns)

    df_res = filter_outliers_by_sliding_window(df)
    pd.testing.assert_frame_equal(df_res.reset_index(drop=True), df_res1.reset_index(drop=True))

    data2 = [
        [1000, 1, 0, 5, 1],
        [1200, 1, 0, 6, 2],
        [1400, 1, 0, 7, 1],
        [1600, 1, 0, 15, 3],
        [1800, 1, 0, 8, 1],
    ]
    res2 = [
        [1000, 1, 0, 5, 1],
        [1800, 1, 0, 8, 1],
    ]
    df = pd.DataFrame(data2, columns=columns)
    df_res2 = pd.DataFrame(res2, columns=columns)

    df_res = filter_outliers_by_sliding_window(df)
    pd.testing.assert_frame_equal(df_res.reset_index(drop=True), df_res2.reset_index(drop=True))

    data3 = [
        [500, 0, 1, 4, 0],
        [750, 0, 1, 4, 0],
        [1000, 0, 1, 20, 0],
        [1250, 0, 1, 5, 1],
    ]
    res3 = [
        [500, 0, 1, 4, 0],
        [750, 0, 1, 4, 0],
        [1250, 0, 1, 5, 1],
    ]
    df = pd.DataFrame(data3, columns=columns)
    df_res3 = pd.DataFrame(res3, columns=columns)

    df_res = filter_outliers_by_sliding_window(df)
    pd.testing.assert_frame_equal(df_res.reset_index(drop=True), df_res3.reset_index(drop=True))

def test_filter_backward_point_jumps():
    data = [
        [100, 0, 0, 3, 4],
        [200, 0, 0, 5, 3],
        [300, 0, 0, 2, 3],
        [400, 0, 0, 0, 0],
        [500, 0, 1, 0, 10],
        [600, 0, 1, 0, 0],
        [700, 0, 1, 0, 11],
        [800, 0, 1, 0, 4],
    ]
    columns = ['frame', 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points']
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        [100, 0, 0, 3, 4],
        [200, 0, 0, 5, 3],
        [300, 0, 0, 2, 3],
        [400, 0, 0, 0, 0],
        [500, 0, 1, 0, 10],
        [700, 0, 1, 0, 11],
    ]
    df_expected = pd.DataFrame(expected_data, columns=columns).reset_index(drop=True)

    df_filtered = filter_backward_point_jumps(df)
    pd.testing.assert_frame_equal(df_filtered, df_expected)

    data = [
        [100, 1, 0, 8, 7],
        [200, 1, 0, 7, 6],
        [300, 1, 0, 2, 6],
        [400, 1, 0, 1, 2],
        [500, 1, 1, 0, 0],
        [600, 1, 1, 0, 0],
    ]
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        [100, 1, 0, 8, 7],
        [200, 1, 0, 7, 6],
        [400, 1, 0, 1, 2],
        [500, 1, 1, 0, 0],
        [600, 1, 1, 0, 0],
    ]
    df_expected = pd.DataFrame(expected_data, columns=columns).reset_index(drop=True)

    df_filtered = filter_backward_point_jumps(df)
    pd.testing.assert_frame_equal(df_filtered, df_expected)

    data = [
        [100, 0, 2, 5, 5],
        [200, 0, 2, 9, 7],
        [300, 0, 2, 5, 3],
        [400, 0, 2, 3, 3],
        [500, 0, 2, 0, 0],
        [600, 0, 2, 0, 0],
    ]
    df = pd.DataFrame(data, columns=columns)

    expected_data = data
    df_expected = pd.DataFrame(expected_data, columns=columns).reset_index(drop=True)

    df_filtered = filter_backward_point_jumps(df)
    pd.testing.assert_frame_equal(df_filtered, df_expected)

    data = [
        [100, 0, 0, 10, 10],
        [200, 0, 0, 8, 5],
        [300, 0, 0, 5, 1],
        [400, 0, 0, 0, 0],
    ]
    df = pd.DataFrame(data, columns=columns)
    expected_data = [
        [100, 0, 0, 10, 10],
        [300, 0, 0, 5, 1],
    ]
    df_expected = pd.DataFrame(expected_data, columns=columns).reset_index(drop=True)

    df_filtered = filter_backward_point_jumps(df, tolerance=4)
    pd.testing.assert_frame_equal(df_filtered, df_expected)

def test_fill_starting_zero_points():
    columns = ['frame', 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points']

    data1 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 3, 2],
        [300, 0, 0, 5, 4],
    ]
    df1 = pd.DataFrame(data1, columns=columns)
    expected_data1 = data1
    df_expected1 = pd.DataFrame(expected_data1, columns=columns).reset_index(drop=True)
    result1 = fill_starting_zero_points(df1).reset_index(drop=True)
    pd.testing.assert_frame_equal(result1, df_expected1)

    data2 = [
        [200, 0, 0, 3, 2],
        [300, 0, 0, 5, 4],
    ]
    df2 = pd.DataFrame(data2, columns=columns)
    expected_data2 = [
        [199, 0, 0, 0, 0],
        [200, 0, 0, 3, 2],
        [300, 0, 0, 5, 4],
    ]
    df_expected2 = pd.DataFrame(expected_data2, columns=columns).reset_index(drop=True)
    result2 = fill_starting_zero_points(df2).reset_index(drop=True)
    pd.testing.assert_frame_equal(result2, df_expected2)

    data3 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 3, 2],
        [300, 1, 0, 2, 1],
        [400, 1, 0, 4, 3],
        [500, 1, 1, 1, 2],
    ]
    df3 = pd.DataFrame(data3, columns=columns)
    expected_data3 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 3, 2],
        [299, 1, 0, 0, 0],
        [300, 1, 0, 2, 1],
        [400, 1, 0, 4, 3],
        [499, 1, 1, 0, 0],
        [500, 1, 1, 1, 2],
    ]
    df_expected3 = pd.DataFrame(expected_data3, columns=columns).reset_index(drop=True)
    result3 = fill_starting_zero_points(df3).reset_index(drop=True)
    pd.testing.assert_frame_equal(result3, df_expected3)

def test_fill_missing_points_inside_sets():
    columns = ['frame', 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points']

    data1 = [
        [100, 0, 0, 0, 0],
        [400, 0, 0, 2, 1],
    ]
    df1 = pd.DataFrame(data1, columns=columns)
    expected_data1 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 1, 0],
        [300, 0, 0, 2, 0],
        [400, 0, 0, 2, 1],
    ]
    df_expected1 = pd.DataFrame(expected_data1, columns=columns)
    result1 = fill_missing_points_inside_sets(df1)
    pd.testing.assert_frame_equal(result1, df_expected1)

    data2 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 1, 0],
        [300, 0, 0, 1, 1],
    ]
    df2 = pd.DataFrame(data2, columns=columns)
    expected_data2 = data2
    df_expected2 = pd.DataFrame(expected_data2, columns=columns)
    result2 = fill_missing_points_inside_sets(df2)
    pd.testing.assert_frame_equal(result2, df_expected2)

    data3 = [
        [100, 0, 0, 0, 0],
        [500, 1, 0, 5, 0],
    ]
    df3 = pd.DataFrame(data3, columns=columns)
    expected_data3 = [
        [100, 0, 0, 0, 0],
        [500, 1, 0, 5, 0],
    ]
    df_expected3 = pd.DataFrame(expected_data3, columns=columns)
    result3 = fill_missing_points_inside_sets(df3)
    pd.testing.assert_frame_equal(result3, df_expected3)

@pytest.mark.parametrize("p1, p2, expected", [
    (11, 0, True),
    (11, 9, True),
    (0, 11, True),
    (15, 13, True),
    (10, 9, False),
    (11, 10, False),
    (8, 3, False),
    (14, 13, False),
])
def test_is_set_finished(p1, p2, expected):
    assert is_set_finished(p1, p2) == expected

def test_complete_sets():
    columns = ['frame', 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points']

    data1 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 11, 5],
    ]
    df1 = pd.DataFrame(data1, columns=columns)
    expected1 = data1
    df_expected1 = pd.DataFrame(expected1, columns=columns)
    result1 = complete_sets(df1)
    pd.testing.assert_frame_equal(result1, df_expected1)

    data2 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 10, 8],
    ]
    df2 = pd.DataFrame(data2, columns=columns)
    expected2 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 10, 8],
        [201, 0, 0, 11, 8],
    ]
    df_expected2 = pd.DataFrame(expected2, columns=columns)
    result2 = complete_sets(df2)
    pd.testing.assert_frame_equal(result2, df_expected2)

    data3 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 10, 9],
        [300, 1, 0, 0, 0],
        [400, 1, 0, 11, 3],
    ]
    df3 = pd.DataFrame(data3, columns=columns)
    expected3 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 10, 9],
        [201, 0, 0, 11, 9],
        [300, 1, 0, 0, 0],
        [400, 1, 0, 11, 3],
    ]
    df_expected3 = pd.DataFrame(expected3, columns=columns)
    result3 = complete_sets(df3)
    pd.testing.assert_frame_equal(result3, df_expected3)

    data4 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 11, 10],
    ]
    df4 = pd.DataFrame(data4, columns=columns)
    expected4 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 11, 10],
        [201, 0, 0, 12, 10],
    ]
    df_expected4 = pd.DataFrame(expected4, columns=columns)
    result4 = complete_sets(df4)
    pd.testing.assert_frame_equal(result4, df_expected4)

    data5 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 10, 10],
    ]
    df5 = pd.DataFrame(data5, columns=columns)
    expected5 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 10, 10],
        [201, 0, 0, 11, 10],
        [202, 0, 0, 12, 10],
    ]
    df_expected5 = pd.DataFrame(expected5, columns=columns)
    result5 = complete_sets(df5)
    pd.testing.assert_frame_equal(result5, df_expected5)

def test_remove_duplicate_scores():
    columns = ['frame', 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points']

    data1 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 1, 0],
        [300, 0, 0, 1, 1],
    ]
    df1 = pd.DataFrame(data1, columns=columns)
    expected1 = data1
    df_expected1 = pd.DataFrame(expected1, columns=columns)
    result1 = remove_duplicate_scores(df1)
    pd.testing.assert_frame_equal(result1, df_expected1)

    data2 = [
        [100, 0, 0, 0, 0],
        [150, 0, 0, 0, 0],
        [200, 0, 0, 1, 0],
        [250, 0, 0, 1, 0],
        [300, 0, 0, 1, 1],
    ]
    df2 = pd.DataFrame(data2, columns=columns)
    expected2 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 1, 0],
        [300, 0, 0, 1, 1],
    ]
    df_expected2 = pd.DataFrame(expected2, columns=columns)
    result2 = remove_duplicate_scores(df2)
    pd.testing.assert_frame_equal(result2, df_expected2)

    data3 = [
        [100, 0, 0, 0, 0],
        [200, 0, 0, 1, 0],
        [300, 0, 0, 0, 0],
    ]
    df3 = pd.DataFrame(data3, columns=columns)
    expected3 = data3
    df_expected3 = pd.DataFrame(expected3, columns=columns)
    result3 = remove_duplicate_scores(df3)
    pd.testing.assert_frame_equal(result3, df_expected3)

@pytest.mark.parametrize(
    "total_points, first_server, expected_server",
    [
        (0, 1, 1),
        (1, 1, 1),
        (4, 1, 1),
        (5, 1, 1),
        (6, 1, 2),
        (23, 1, 2),
        (0, 2, 2),
        (1, 2, 2),
        (2, 2, 1),
        (3, 2, 1),
        (24, 2, 2),
        (25, 2, 1),
    ],
)
def test_deduce_server(total_points, first_server, expected_server):
    assert deduce_server(total_points, first_server) == expected_server

@pytest.mark.parametrize(
    "input_data,expected_server,expected_won,expected_rallie_time",
    [
        (
            [
                {"frame": 0, "player_1_points": 0, "player_2_points": 0, "player_1_sets": 0, "player_2_sets": 0},
                {"frame": 30, "player_1_points": 1, "player_2_points": 0, "player_1_sets": 0, "player_2_sets": 0},
                {"frame": 60, "player_1_points": 1, "player_2_points": 1, "player_1_sets": 0, "player_2_sets": 0},
                {"frame": 90, "player_1_points": 2, "player_2_points": 1, "player_1_sets": 0, "player_2_sets": 0},
                {"frame": 120, "player_1_points": 2, "player_2_points": 2, "player_1_sets": 0, "player_2_sets": 0},
            ],
            [1, 1, 2, 2, 1],
            [1, 2, 1, 2, pd.NA],
            [1.0, 1.0, 1.0, 1.0, pd.NA],
        ),
    ]
)
def test_add_match_info(input_data, expected_server, expected_won, expected_rallie_time):
    df_input = pd.DataFrame(input_data)
    df_result = add_match_info(df_input)

    assert df_result["server"].tolist() == expected_server
    assert df_result["won"].tolist() == expected_won
    assert df_result["rallie_time"].tolist() == expected_rallie_time