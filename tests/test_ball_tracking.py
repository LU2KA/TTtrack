import pytest
from src.tt_track.ball_tracking.ball_tracking import detect_bounces, linear_fill_trajectory


@pytest.mark.parametrize("test_case", [
    {
        "name": "multiple_bounces",
        "trajectory": [
            {"frame": 1, "cx": 1200, "cy": 400},
            {"frame": 2, "cx": 1200, "cy": 450},
            {"frame": 3, "cx": 1200, "cy": 400},
            {"frame": 4, "cx": 1200, "cy": 350},
            {"frame": 5, "cx": 1200, "cy": 450},
            {"frame": 6, "cx": 1200, "cy": 500},
            {"frame": 7, "cx": 1200, "cy": 450},
        ],
        "expected": [
            {"frame": 2, "cx": 1200, "cy": 450},
            {"frame": 6, "cx": 1200, "cy": 500}
        ]
    },
    {
        "name": "no_motion",
        "trajectory": [
            {"frame": 1, "cx": 1200, "cy": 400},
            {"frame": 2, "cx": 1200, "cy": 400},
            {"frame": 3, "cx": 1200, "cy": 400},
        ],
        "expected": []
    },
    {
        "name": "horizontal_motion",
        "trajectory": [
            {"frame": 1, "cx": 1000, "cy": 400},
            {"frame": 2, "cx": 1100, "cy": 400},
            {"frame": 3, "cx": 1200, "cy": 400},
        ],
        "expected": []
    },
    {
        "name": "small_vertical_motion",
        "trajectory": [
            {"frame": 1, "cx": 1200, "cy": 400},
            {"frame": 2, "cx": 1200, "cy": 401},
            {"frame": 3, "cx": 1200, "cy": 400},
        ],
        "expected": []
    },
    {
        "name": "single_frame_bounce",
        "trajectory": [
            {"frame": 1, "cx": 1200, "cy": 400},
            {"frame": 2, "cx": 1200, "cy": 450},
            {"frame": 3, "cx": 1200, "cy": 400},
        ],
        "expected": [{"frame": 2, "cx": 1200, "cy": 450}]
    },
    {
        "name": "noisy_data",
        "trajectory": [
            {"frame": 1, "cx": 1200, "cy": 400},
            {"frame": 2, "cx": 1200, "cy": 405},
            {"frame": 3, "cx": 1200, "cy": 410},
            {"frame": 4, "cx": 1200, "cy": 450},
            {"frame": 5, "cx": 1200, "cy": 410},
            {"frame": 6, "cx": 1200, "cy": 405},
            {"frame": 7, "cx": 1200, "cy": 400},
        ],
        "expected": [{"frame": 4, "cx": 1200, "cy": 450}]
    },
    {
        "name": "empty_input",
        "trajectory": [],
        "expected": []
    },
    {
        "name": "single_point",
        "trajectory": [{"frame": 1, "cx": 1200, "cy": 400}],
        "expected": []
    },
    {
        "name": "diagonal_motion",
        "trajectory": [
            {"frame": 1, "cx": 1000, "cy": 400},
            {"frame": 2, "cx": 1200, "cy": 450},
            {"frame": 3, "cx": 1300, "cy": 400},
        ],
        "expected": [{"frame": 2, "cx": 1200, "cy": 450}]
    },
    {
        "name": "noisy_data_with_different_position",
        "trajectory": [
            {"frame": 1, "cx": 2200, "cy": 400},
            {"frame": 2, "cx": 2200, "cy": 405},
            {"frame": 3, "cx": 2200, "cy": 410},
            {"frame": 4, "cx": 2200, "cy": 450},
            {"frame": 5, "cx": 2200, "cy": 410},
            {"frame": 6, "cx": 2200, "cy": 405},
            {"frame": 7, "cx": 2200, "cy": 400},
        ],
        "expected": []
    }
])
def test_detect_bounces(test_case):
    bounces = detect_bounces(test_case["trajectory"])

    assert len(bounces) == len(test_case["expected"])

    for actual_bounce, expected_bounce in zip(bounces, test_case["expected"]):
        assert actual_bounce["frame"] == expected_bounce["frame"]
        assert actual_bounce["cx"] == expected_bounce["cx"]
        assert actual_bounce["cy"] == expected_bounce["cy"]


@pytest.mark.parametrize("test_case", [
    {
        "name": "empty_input",
        "input": [],
        "expected": []
    },
    {
        "name": "single_frame",
        "input": [{"frame": 1, "cx": 100, "cy": 200}],
        "expected": [{"frame": 1, "cx": 100, "cy": 200}]
    },
    {
        "name": "consecutive_frames_no_gaps",
        "input": [
            {"frame": 1, "cx": 100, "cy": 200},
            {"frame": 2, "cx": 110, "cy": 210},
            {"frame": 3, "cx": 120, "cy": 220}
        ],
        "expected": [
            {"frame": 1, "cx": 100, "cy": 200},
            {"frame": 2, "cx": 110, "cy": 210},
            {"frame": 3, "cx": 120, "cy": 220}
        ]
    },
    {
        "name": "small_gap_filled",
        "input": [
            {"frame": 1, "cx": 100, "cy": 200},
            {"frame": 4, "cx": 130, "cy": 230}
        ],
        "expected": [
            {"frame": 1, "cx": 100, "cy": 200},
            {"frame": 2, "cx": 110.0, "cy": 210.0, "x1": 100, "y1": 200, "x2": 120, "y2": 220, "confidence": 0.0},
            {"frame": 3, "cx": 120.0, "cy": 220.0, "x1": 110, "y1": 210, "x2": 130, "y2": 230, "confidence": 0.0},
            {"frame": 4, "cx": 130, "cy": 230}
        ]
    },
    {
        "name": "with_additional_fields",
        "input": [
            {"frame": 1, "cx": 100, "cy": 200, "x1": 90, "y1": 190, "x2": 110, "y2": 210, "confidence": 0.9},
            {"frame": 3, "cx": 120, "cy": 220, "x1": 110, "y1": 210, "x2": 130, "y2": 230, "confidence": 0.8}
        ],
        "expected": [
            {"frame": 1, "cx": 100, "cy": 200, "x1": 90, "y1": 190, "x2": 110, "y2": 210, "confidence": 0.9},
            {"frame": 2, "cx": 110.0, "cy": 210.0, "x1": 100, "y1": 200, "x2": 120, "y2": 220, "confidence": 0.0},
            {"frame": 3, "cx": 120, "cy": 220, "x1": 110, "y1": 210, "x2": 130, "y2": 230, "confidence": 0.8}
        ]
    },
    {
        "name": "non_consecutive_with_fillable_gaps",
        "input": [
            {"frame": 1, "cx": 100, "cy": 200},
            {"frame": 3, "cx": 120, "cy": 220},
            {"frame": 5, "cx": 140, "cy": 240}
        ],
        "expected": [
            {"frame": 1, "cx": 100, "cy": 200},
            {"frame": 2, "cx": 110.0, "cy": 210.0, "x1": 100, "y1": 200, "x2": 120, "y2": 220, "confidence": 0.0},
            {"frame": 3, "cx": 120, "cy": 220},
            {"frame": 4, "cx": 130.0, "cy": 230.0, "x1": 120, "y1": 220, "x2": 140, "y2": 240, "confidence": 0.0},
            {"frame": 5, "cx": 140, "cy": 240}
        ]
    }
])
def test_linear_fill_trajectory(test_case):
    result = linear_fill_trajectory(test_case["input"])

    assert len(result) == len(test_case["expected"])

    for res, exp in zip(result, test_case["expected"]):
        assert res["frame"] == exp["frame"]
        assert res["cx"] == pytest.approx(exp["cx"])
        assert res["cy"] == pytest.approx(exp["cy"])

        for key in exp:
            if key not in ["frame", "cx", "cy"]:
                if key in res:
                    assert res[key] == pytest.approx(exp[key]) if isinstance(exp[key], float) else res[key] == exp[key]
                else:
                    pytest.fail(f"Missing expected key {key} in result")