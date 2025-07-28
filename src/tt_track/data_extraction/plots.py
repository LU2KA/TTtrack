"""
This module is used in combination with extract_match_data.py it is used to visualize the read data.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Similar graph I used during BI-ZUM, so I reused it here as well
def create_colormaps():
    """
    Create and return blue and red gradient colormaps.

    :return: Tuple (blue_cmap, red_cmap) of matplotlib LinearSegmentedColormap objects.
    """
    blue_colors = ["#E3F2FD", "#2196F3", "#0D47A1"]
    red_colors = ["#FFEBEE", "#F44336", "#B71C1C"]
    blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", blue_colors)
    red_cmap = LinearSegmentedColormap.from_list("red_gradient", red_colors)
    return blue_cmap, red_cmap


def get_color(value, zero, max_val, blue_cmap, red_cmap):
    """
    Compute a color for a given value relative to a baseline and max value.

    Values above the baseline use the red colormap; below use the blue colormap.

    :param value: Numeric value to map to a color.
    :param zero: Baseline (zero) value for comparison.
    :param max_val: Maximum absolute difference to normalize intensity.
    :param blue_cmap: Blue colormap for values below zero.
    :param red_cmap: Red colormap for values above or equal to zero.
    :return: RGBA tuple color.
    """
    intensity = min(abs(value - zero) / max_val, 1.0)
    if value >= zero:
        return red_cmap(intensity)
    return blue_cmap(intensity)


def plot_segments(ax, x, y, zero=0):
    """
    Plot (x, y) as colored line segments and points indicating deviation from zero.

    Colors reflect the magnitude and direction of y relative to zero, blending
    smoothly without special zero-crossing handling.

    :param ax: Matplotlib Axes to plot on.
    :param x: Sequence of x coordinates.
    :param y: Sequence of y coordinates.
    :param zero: Baseline value for coloring reference (default 0).
    """
    x, y = np.asarray(x), np.asarray(y)
    max_diff = np.max(np.abs(y - zero)) or 1
    blue_cmap, red_cmap = create_colormaps()

    n = len(x)
    if n == 0:
        return

    for j in range(n - 1):
        avg_val = (y[j] + y[j + 1]) / 2
        color = get_color(avg_val, zero, max_diff, blue_cmap, red_cmap)
        ax.plot([x[j], x[j + 1]], [y[j], y[j + 1]], color=color, linewidth=2)
        ax.plot(x[j], y[j], "o", color=color, markersize=6)

    last_color = get_color(y[-1], zero, max_diff, blue_cmap, red_cmap)
    ax.plot(x[-1], y[-1], "o", color=last_color, markersize=6)


def plot_all_sets_line_graph(df, player1_name="Player 1", player2_name="Player 2"):
    """
    Plot line graphs of points difference for all sets in the match.

    Each subplot corresponds to one set, showing the point difference progression.
    X-axis labels show the score at each point.

    :param df: DataFrame with columns 'player_1_sets', 'player_2_sets', 'player_1_points', 'player_2_points'.
    :param player1_name: Display name for Player 1.
    :param player2_name: Display name for Player 2.
    :return: Matplotlib figure object containing all set plots.
    """
    df_plot = df.copy()

    df_plot["set"] = (
        df_plot["player_1_sets"].astype(str)
        + "-"
        + df_plot["player_2_sets"].astype(str)
    )
    df_plot["points_diff"] = df_plot["player_1_points"] - df_plot["player_2_points"]
    df_plot["score_label"] = (
        df_plot["player_1_points"].astype(str)
        + ":"
        + df_plot["player_2_points"].astype(str)
    )

    unique_sets = df_plot["set"].unique()
    n = len(unique_sets)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 12, rows * 6), constrained_layout=True
    )
    axs = axs.flatten() if n > 1 else [axs]

    for ax, s in zip(axs, unique_sets):
        subset = df_plot[df_plot["set"] == s]
        unique_scores = subset["score_label"].unique()
        y = subset.set_index("score_label").loc[unique_scores, "points_diff"].values

        ax.axhline(0, color="black", linewidth=2, linestyle="--")
        plot_segments(ax, np.arange(len(unique_scores)), y)

        ax.set_ylim(
            -max(abs(y.min()), abs(y.max())) - 1, max(abs(y.min()), abs(y.max())) + 1
        )

        ax.set_title(f"Set {s}")
        ax.set_xlabel(f"Score ({player1_name} : {player2_name})")
        ax.set_ylabel("Points Difference")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(len(unique_scores)))
        ax.set_xticklabels(unique_scores)

    for ax in axs[n:]:
        ax.axis("off")

    return fig


def plot_serve_win_percentages(df, player1_name="Player 1", player2_name="Player 2"):
    """
    Plot pie charts showing point win percentages when each player serves.

    :param df: DataFrame with columns 'server', 'player_1_points', 'player_2_points'.
    :param player1_name: Display name for Player 1.
    :param player2_name: Display name for Player 2.
    :return: Matplotlib figure object with two pie charts.
    """
    df = df.copy()
    df[["player_1_point_delta", "player_2_point_delta"]] = (
        df[["player_1_points", "player_2_points"]].diff().fillna(0)
    )

    p1_serves = df[df["server"] == 1]
    p2_serves = df[df["server"] == 2]

    colors = "#F44336", "#2196F3"

    p1_wins = [
        (p1_serves["player_1_point_delta"] > 0).sum(),
        (p1_serves["player_2_point_delta"] > 0).sum(),
    ]

    p2_wins = [
        (p2_serves["player_1_point_delta"] > 0).sum(),
        (p2_serves["player_2_point_delta"] > 0).sum(),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].pie(
        p1_wins,
        labels=[f"{player1_name} won", f"{player2_name} won"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    axs[0].set_title(f"When {player1_name} is Serving")

    axs[1].pie(
        p2_wins,
        labels=[f"{player1_name} won", f"{player2_name} won"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    axs[1].set_title(f"When {player2_name} is Serving")

    return fig


def plot_rally_durations(df, player1_name="Player 1", player2_name="Player 2"):
    """
    Plot density plots of rally durations overall, by winner, and by server.

    :param df: DataFrame with 'rallie_time', 'won', and 'server' columns.
    :param player1_name: Display name for Player 1.
    :param player2_name: Display name for Player 2.
    :return: Matplotlib figure with three subplots.
    """
    colors = {1: "#F44336", 2: "#2196F3"}

    df_valid = df.dropna(subset=["rallie_time"])

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    sns.kdeplot(
        data=df_valid,
        x="rallie_time",
        ax=axs[0],
        fill=True,
        color="gray",
        linewidth=2,
        alpha=0.4,
        label="All Rallies",
    )
    axs[0].set_title("Density Plot of Rally Duration (All)")
    axs[0].set_ylabel("Density")
    axs[0].grid(True, alpha=0.3)

    for winner, group in df_valid.dropna(subset=["won"]).groupby("won"):
        label = f"{player1_name} won" if winner == 1 else f"{player2_name} won"
        sns.kdeplot(
            data=group,
            x="rallie_time",
            ax=axs[1],
            fill=True,
            linewidth=2,
            alpha=0.5,
            label=label,
            color=colors.get(winner, "gray"),
        )
    axs[1].set_title("Density Plot of Rally Duration by Winner")
    axs[1].set_ylabel("Density")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    for server, group in df_valid.dropna(subset=["server"]).groupby("server"):
        label = f"{player1_name} served" if server == 1 else f"{player2_name} served"
        sns.kdeplot(
            data=group,
            x="rallie_time",
            ax=axs[2],
            fill=True,
            linewidth=2,
            alpha=0.5,
            label=label,
            color=colors.get(server, "gray"),
        )
    axs[2].set_title("Density Plot of Rally Duration by Server")
    axs[2].set_xlabel("Rally Duration (seconds)")
    axs[2].set_ylabel("Density")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    for ax in axs:
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlim(left=0)

    return fig


def plot_endgame_points(df, player1_name="Player 1", player2_name="Player 2"):
    """
    Plot a pie chart showing which player won points during endgame phases,
    defined as points where both players have at least 8 points.

    Endgame points typically represent the critical closing phase of a set.

    :param df: pandas DataFrame containing match point data with columns
               including 'player_1_points', 'player_2_points', and 'won' (indicating the point winner).
    :param player1_name: Display name for Player 1 (default is "Player 1").
    :param player2_name: Display name for Player 2 (default is "Player 2").
    :return: matplotlib.figure.Figure object containing the pie chart.
    """
    deuce_df = df[(df["player_1_points"] >= 8) & (df["player_2_points"] >= 8)]
    counts = deuce_df["won"].value_counts()

    # Using this way to get labels, because sometimes only one player wins all these points
    label_map = {1: player1_name, 2: player2_name}
    labels = [label_map.get(player, f"Player {player}") for player in counts.index]

    colors = ["#F44336", "#2196F3"]

    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("Endgame Points Won (Both Players ≥8)")

    return fig
