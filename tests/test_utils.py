import shutil
import subprocess

from src.tt_track.utils.preprocessing import download_youtube_video, cut_video, reencode_video, make_tournament_folder
import pytest
from pathlib import Path
import time

from pathlib import Path

import subprocess
from pathlib import Path


TEST_VIDEO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
SHORT_URL = "https://youtu.be/jNQXAC9IVRw"


def test_download_youtube_video_all(tmp_path):
    non_existent_folder = tmp_path / "nonexistent_folder"
    with pytest.raises(ValueError, match="The output folder does not exist"):
        download_youtube_video(TEST_VIDEO_URL, output_folder=non_existent_folder)
    assert not non_existent_folder.exists()

    download_youtube_video(TEST_VIDEO_URL, output_folder=str(tmp_path))
    files = list(tmp_path.glob("match_*.mp4"))
    assert len(files) == 1
    assert files[0].stat().st_size > 0
    for f in files:
        f.unlink()

    download_youtube_video(SHORT_URL, output_folder=str(tmp_path))
    files = list(tmp_path.glob("match_*.mp4"))
    assert len(files) == 1
    for f in files:
        f.unlink()

    custom_name = "test_video_output"
    download_youtube_video(TEST_VIDEO_URL, output_name=custom_name, output_folder=str(tmp_path))
    output_file = tmp_path / f"{custom_name}.mp4"
    assert output_file.exists()
    assert output_file.stat().st_size > 0



@pytest.mark.parametrize("invalid_url", [
    "not_a_url",
    "https://notyoutube.com/video",
    "https://www.youtube.com",
    "https://www.youtube.com/playlist?list=invalid",
    "https://www.youtube.com/watch?v=private_video_test",
    "https://www.youtube.com/watch?v=aaaaaaaaaaa",
    None,
    12345,
    {"invalid": "input"},

])
def test_various_invalid_inputs(invalid_url):
    with pytest.raises(ValueError):
        download_youtube_video(invalid_url)



def test_cut_video(tmp_path):
    def get_video_duration(file_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)

    input_file = Path(__file__).parent / "data" / "test.mp4"
    assert input_file.exists(), "Test video file does not exist"

    output_file = tmp_path / "output.mp4"
    cut_video(start_time=0, end_time=30, input_file=str(input_file), output_file=str(output_file))

    assert output_file.exists(), "Output file was not created"
    duration = get_video_duration(output_file)
    assert 29 <= duration <= 31, f"Cut video duration is off: {duration}s"

    output_file = tmp_path / "output1.mp4"
    cut_video(start_time=20, end_time=80, input_file=str(input_file), output_file=str(output_file))

    assert output_file.exists(), "Output file was not created"
    duration = get_video_duration(output_file)
    assert 59 <= duration <= 61, f"Cut video duration is off: {duration}s"

    with pytest.raises(FileNotFoundError):
        cut_video(start_time=0, end_time=10, input_file="nonexistent.mp4", output_file=str(tmp_path / "out.mp4"))

    with pytest.raises(ValueError):
        cut_video(start_time=10, end_time=5, input_file=str(input_file), output_file=str(tmp_path / "out2.mp4"))

    with pytest.raises(ValueError):
        cut_video(start_time=5, end_time=5, input_file=str(input_file), output_file=str(tmp_path / "out3.mp4"))


def test_reencode_video(tmp_path):
    input_file = Path(__file__).parent / "data" / "test.mp4"
    output_file = tmp_path / "reencoded_output.mp4"

    reencode_video(input_file, str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0
    input_file = "nonexistent_input.mp4"
    output_file = tmp_path / "reencoded_output.mp4"

    with pytest.raises(subprocess.CalledProcessError):
        reencode_video(input_file, str(output_file))

def test_make_tournament_folder_creates_structure():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    base, title = make_tournament_folder(url)

    assert base.exists()

    assert (base / "raw").is_dir()
    assert (base / "cuts").is_dir()
    assert (base / "analysis").is_dir()
    assert title

    if base.exists():
        shutil.rmtree(base)
