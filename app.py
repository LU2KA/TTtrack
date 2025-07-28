"""
User interface for my app TT Track
"""
import os
import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st

from src.tt_track.ball_tracking.ball_tracking import (
    process_video_and_get_positions,
    linear_fill_trajectory,
    detect_bounces,
)
from src.tt_track.utils.preprocessing import (
    download_youtube_video,
    cut_video,
    reencode_video,
    make_tournament_folder,
)
from src.tt_track.data_extraction.extract_match_data import (
    extract_match_segments,
    get_latest_file,
    process_video_data,
    results_to_summary_dataframe,
    correcting_points_dataset,
    add_match_info,
)
from src.tt_track.data_extraction.plots import (
    plot_all_sets_line_graph,
    plot_rally_durations,
    plot_serve_win_percentages,
    plot_endgame_points,
)
from src.tt_track.ball_tracking.vizualization import (
    map_and_plot_regional_heatmap,
    map_and_plot_bounces_topdown,
    display_ball_trajectory_with_collisions_mp4,
)

st.set_page_config(page_title="TT Track", page_icon="🏓")

@st.cache_data(show_spinner=True)
def get_processed_data(video_file):
    """
    Just caching data from previous function.
    :param video_file
    :return: df
    """
    df = process_video_data(video_file)
    df = results_to_summary_dataframe(df)
    df = correcting_points_dataset(df)
    df = add_match_info(df)
    return df

@st.cache_data(show_spinner=True)
def cached_process_positions(cut_file):
    """
    Just caching data from previous function.

    :param cut_file:
    :return:
    """
    return process_video_and_get_positions(cut_file)
@st.cache_data(show_spinner=False)
def cached_extract_match_segments(video_path, background_path, interval_sec):
    """
    Just caching data from previous function.

    :param video_path:
    :param background_path:
    :param interval_sec:
    :return:
    """
    return extract_match_segments(video_path, background_path, interval_sec)


def save_plot(fig, path: Path, formats=None):
    """
    Just used to save into png and svg.
    :param fig: matplotlib figure object
    :param path: Path object for file name without extension
    :param formats: list of formats to save, e.g., ['png', 'svg']
    """
    if formats is None:
        formats = ["png", "svg"]

    for fmt in formats:
        fig_path = path.with_suffix(f".{fmt}")
        fig.savefig(fig_path, bbox_inches='tight')

def check_file_permissions():
    """
    Function to check file permissions.
    :return: bool
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w", encoding="utf-8") as files:
                files.write("test")
            with open(test_file, "r", encoding="utf-8") as files:
                _ = files.read()
        return True
    except (PermissionError, OSError) as e:
        st.error(f"File access error: {e}")
        return False

for key in ("page","url","cut_result","selected_match_idx","frame_range","approved_range","tournament_base"):
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.page = st.session_state.page or "input"

def reset():
    """
    Resets all session variables.
    :return:
    """
    for k in ("page","url","cut_result","selected_match_idx","frame_range","approved_range","tournament_base"):
        st.session_state[k] = None

if st.session_state.page == "input":
    st.title("🏓 Select a Table Tennis Tournament to Analyze")
    st.markdown("[From YouTube @ttcupczech2](https://www.youtube.com/@ttcupczech2)")
    if not check_file_permissions():
        st.stop()
    url = st.text_input("YouTube URL", key="url_input")
    if url:
        st.session_state.url = url
        st.video(url)
        if st.button("Cut Tournament"):
            base, title = make_tournament_folder(url)
            st.session_state.tournament_base = base
            st.session_state.video_title = title
            st.session_state.page = "cutting"
            st.rerun()

elif st.session_state.page == "cutting":
    st.title("Processing Video…")
    base = st.session_state.tournament_base
    raw_dir = base / "raw"
    cuts_dir = base / "cuts"

    with st.spinner("Downloading & cutting…"):
        try:
            download_youtube_video(st.session_state.url, output_folder=raw_dir)
            video = get_latest_file(raw_dir)
            df_matches = cached_extract_match_segments(video, "./data/background.jpg", interval_sec=30)
            cuts = []
            for i, m in enumerate(df_matches):
                start, end = m["match_start_time_sec"], m["match_end_time_sec"]
                outp = cuts_dir / f"{i}.mp4"
                cut_video(start, end, video, output_file=outp)
                cuts.append(m)
            st.session_state.cut_result = ("success", cuts)
        except (OSError, ValueError, subprocess.SubprocessError, requests.RequestException) as e:
            st.session_state.cut_result = ("error", str(e))

    st.session_state.page = "done"
    st.rerun()

elif st.session_state.page == "done":
    st.title("✅ Cutting Complete")
    status, matches = st.session_state.cut_result or ("error", None)
    if status == "success":
        st.success(f"{len(matches)} matches ready.")
        opts = [f"{i}: {m['player1_name']} vs {m['player2_name']}" for i,m in enumerate(matches)]
        sel = st.selectbox("Pick a match to analyze", opts)
        if len(opts) > 0:
            if st.button("Analyze"):
                st.session_state.selected_match_idx = int(sel.split(":")[0])
                st.session_state.page = "analyze"
                st.rerun()
    else:
        st.error("Cutting failed: " + matches)

    if st.button("Run another"):
        reset()
        st.rerun()

elif st.session_state.page == "analyze":
    st.title("🏓 Match Analysis")
    idx = st.session_state.selected_match_idx
    status, matches = st.session_state.cut_result
    base = st.session_state.tournament_base
    cuts_dir = base / "cuts"
    analysis_dir = base / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    if status=="success":
        match = matches[idx]
        cut_file_save = cuts_dir / f"{idx}.mp4"
        st.markdown(f"### 🎾 {match['player1_name']} vs {match['player2_name']}")
        st.video(cut_file_save)

        plot_base = analysis_dir / f"{idx}"

        df_match = get_processed_data(cut_file_save)
        st.markdown("#### 🎯 Score Progression")
        fig0 = plot_all_sets_line_graph(df_match, match['player1_name'], match['player2_name'])
        st.pyplot(fig0)
        save_plot(fig0, analysis_dir / f"{idx}_score_progression")

        st.markdown("#### ⏱️ Rally Durations")
        fig1 = plot_rally_durations(df_match, match['player1_name'], match['player2_name'])
        st.pyplot(fig1)
        save_plot(fig1, analysis_dir / f"{idx}_rally_durations")

        st.markdown("#### 💥 Serve Wins")
        fig2 = plot_serve_win_percentages(df_match, match['player1_name'], match['player2_name'])
        st.pyplot(fig2)
        save_plot(fig2, analysis_dir / f"{idx}_serve_win_percentages")

        st.markdown("#### 🚀 Endgame points won")
        fig3 = plot_endgame_points(df_match, match['player1_name'], match['player2_name'])
        st.pyplot(fig3)
        save_plot(fig3, analysis_dir / f"{idx}_crucial_points")

        st.markdown("#### 🔎 Range for Ball Tracking")
        lo, hi = int(df_match.index.min()), int(df_match.index.max())

        if st.session_state.frame_range is None:
            st.session_state.frame_range = (lo, hi)
        r = st.slider("Frame range", lo, hi, st.session_state.frame_range, 1)
        st.session_state.frame_range = r

        if st.button("✅ Apply Range"):
            st.session_state.approved_range = r

        if st.session_state.approved_range:
            a,b = st.session_state.approved_range
            sub = df_match.loc[a:b]
            a, b = st.session_state.approved_range
            sub = df_match.loc[a:b]
            st.markdown(f"#### 📄 Rally Data ({a}–{b})")
            st.dataframe(sub)

            fps = 30
            start_sec = df_match.loc[a, 'frame'] / fps
            end_sec = df_match.loc[b, 'frame'] / fps

            dyn_cut = analysis_dir / f"slice_{a}_{b}.mp4"
            cut_video(start_sec, end_sec, cut_file_save, output_file=dyn_cut)

            final_dyn = analysis_dir / f"slice_{a}_{b}_h264.mp4"

            # I struggled with re-encoding the video because the output was saved as MP4, but it would not play in Streamlit.
            reencode_video(dyn_cut, final_dyn)

            ball_df, md = cached_process_positions(final_dyn)
            ball_df = linear_fill_trajectory(ball_df)
            bounces_df = detect_bounces(ball_df)
            annotated_raw = analysis_dir / f"slice_{a}_{b}_annotated_raw.mp4"
            display_ball_trajectory_with_collisions_mp4(
                final_dyn,
                ball_df,
                bounces_df,
                output_path=annotated_raw
            )

            annotated_final = analysis_dir / f"slice_{a}_{b}_annotated.mp4"
            reencode_video(annotated_raw, annotated_final)

            st.markdown("#### 🎥 Annotated & Re‐encoded Slice")
            with open(annotated_final, "rb") as f:
                st.video(f.read())

            st.markdown("#### 🗺️ Heatmap")
            fig5 = map_and_plot_regional_heatmap(bounces_df)
            st.pyplot(fig5)
            save_plot(fig5, analysis_dir / f"{idx}_heatmap")

            st.markdown("#### 📍 Top-down")
            fig6 = map_and_plot_bounces_topdown(bounces_df, table_image_path=Path('./data/table.png'))
            st.pyplot(fig6)
            save_plot(fig6, analysis_dir / f"{idx}_topdown")


    else:
        st.error("No match selected or error.")

    if st.button("⬅️ Back"):
        for k_end in ("frame_range","approved_range"):
            st.session_state.pop(k_end, None)
        st.session_state.page = "done"
        st.rerun()
