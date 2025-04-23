"""GUI for inferencing interactive segmentation models."""

import matplotlib

matplotlib.use("Agg")

import argparse
import tkinter as tk

import torch

from core.inference.utils import find_checkpoint, load_is_model
from core.interactive_demo.app import InteractiveDemoApp
from core.utils.exp import load_config_file


def main():
    args, cfg = parse_args()

    torch.backends.cudnn.deterministic = True
    checkpoint_path = find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = load_is_model(
        checkpoint_path, args.device, args.eval_ritm, cpu_dist_maps=True
    )

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the checkpoint. "
        "This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) "
        "or an absolute path. The file extension can be omitted.",
    )

    parser.add_argument("--gpu", type=int, default=0, help="Id of GPU to use.")

    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Use only CPU for inference."
    )

    parser.add_argument(
        "--limit-longest-size",
        type=int,
        default=800,
        help="If the largest side of an image exceeds this value, "
        "it is resized so that its largest side is equal to this value.",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/main_cfg.yaml",
        help="The path to the config file.",
    )

    parser.add_argument("--eval-ritm", action="store_true", default=False)

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}")
    cfg = load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == "__main__":
    main()
