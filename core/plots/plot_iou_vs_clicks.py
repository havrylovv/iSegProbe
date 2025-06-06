"""Plot mIoU vs Clicks from pre-computed Pickle Logs."""

import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_iou_vs_clicks(
    pickle_paths: List[str],
    legend_names: List[str],
    output_folder: str,
    title: str = "",
) -> None:
    """Generates a plot of mean Intersection over Union (mIoU) over a series of clicks from multiple pickle files.
    It reads the mIoU data from the provided pickle files, computes the mean mIoU for each click, and plots the results.
    Pickle files are generated by the evaluation script.
    """
    if len(pickle_paths) != len(legend_names):
        raise ValueError("Number of paths must match number of legend names")

    os.makedirs(output_folder, exist_ok=True)

    # Different markers for distinction
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "*",
        "X",
    ]

    plt.figure(figsize=(10, 6))

    for idx, path in tqdm(
        enumerate(pickle_paths), total=len(pickle_paths), desc="Processing Pickle Files"
    ):
        with open(path, "rb") as f:
            data = pickle.load(f)

        stacked = np.stack(data)
        mean_iou = np.mean(stacked, axis=0)
        num_clicks = np.arange(1, len(mean_iou) + 1)

        plt.plot(
            num_clicks,
            mean_iou * 100,
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=legend_names[idx],
        )

    plt.xlabel("Number of Clicks", fontsize=14)
    plt.ylabel("mIoU (%)", fontsize=14)

    # Show only every second number on X-axis
    plt.xticks(num_clicks[::2])

    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.title(title, fontsize=18)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"iou_time_series_{current_time}.png")
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")


def parse_args():
    parser = ArgumentParser(
        description="Plot mIoU vs Clicks from pre-computed Pickle Logs."
    )
    parser.add_argument(
        "--pickle_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to pickle files containing click-wise mIoU data. Pickle files are obtained from the evaluation by setting `save_ious` to True in evaluate.py",
    )
    parser.add_argument(
        "--legend_names",
        type=str,
        nargs="+",
        required=True,
        help="Legend labels corresponding to each pickle file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the output plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional title for the plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_iou_vs_clicks(
        pickle_paths=args.pickle_paths,
        legend_names=args.legend_names,
        output_folder=args.output_folder,
        title=args.title,
    )


if __name__ == "__main__":
    main()
