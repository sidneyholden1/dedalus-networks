"""Compose the Poisson and KdV images used in the repository README.

This only changes the presentation of the saved figure, using Matplotlib,
NumPy and Pillow. It does not solve or interpolate the KdV trajectory.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


repository = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source", type=Path,
    default=repository / "docs/pages/examples/images/network_kdv.png",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--poisson-source", type=Path,
    default=repository / "docs/pages/examples/images/network_poisson.png",
)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
if source_sha256 != "84144303c3c049aee2b6b0457f144eb9483d4715dab8e625e5ae363b1bc1db9d":
    raise ValueError("Expected the retained four-panel paper figure")
with Image.open(args.source) as image:
    image.load()
    if image.size != (1928, 1623):
        raise ValueError("Unexpected source dimensions")
    # The same square viewports used in the reviewed README preview.
    source_bounds = [(240, 86), (976, 86), (240, 889), (976, 889)]
    panels = [np.asarray(image.crop((x, y, x + 718, y + 718)).convert("RGB"))
              for x, y in source_bounds]

# Logical layout in displayed pixels; export at twice that resolution.
width, height = 790, 244
side, gap, top = 182.5, 20, 25
times = (0, 6, 20, 200)
outputs = {}
# Match the heat animation's 5:4 canvas without stretching the Poisson plot.
with Image.open(args.poisson_source) as poisson:
    poisson = poisson.convert("RGB")
    canvas_height = round(poisson.width * 4 / 5)
    if canvas_height < poisson.height:
        raise ValueError("Poisson image is taller than the shared canvas")
    padding = (canvas_height - poisson.height) // 2
    canvas = Image.new("RGB", (poisson.width, canvas_height), "white")
    canvas.paste(poisson, (0, padding))
    destination = args.output / "network_poisson_readme.png"
    canvas.save(destination)
    outputs[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"}):
    for theme, ink in (("light", "#1f2328"), ("dark", "#f0f6fc")):
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=200)
        fig.patch.set_alpha(0)
        for i, (panel, time) in enumerate(zip(panels, times)):
            left = i * (side + gap)
            axis = fig.add_axes((left / width, (height - top - side) / height,
                                 side / width, side / height))
            axis.imshow(panel, interpolation="lanczos")
            axis.set_axis_off()
            fig.text((left + side / 2) / width, 1 - 11 / height,
                     rf"$t = {time}$", ha="center", va="center",
                     fontsize=14 * 72 / 100, color=ink)

        # Swatches identify signs without spelling out colour names.
        legend_y = 229
        for x, colour, label in ((240, "#ffdd72", r"$u > 0$"),
                                 (327, "#5cdef8", r"$u < 0$")):
            fig.add_artist(Rectangle((x / width, 1 - (legend_y + 5) / height),
                                     10 / width, 10 / height,
                                     transform=fig.transFigure,
                                     facecolor=colour, edgecolor="none"))
            fig.text((x + 16) / width, 1 - legend_y / height, label,
                     ha="left", va="center", fontsize=13 * 72 / 100, color=ink)
        fig.text(414 / width, 1 - legend_y / height,
                 r"$\mathrm{Width} \propto |u|$", ha="left", va="center",
                 fontsize=13 * 72 / 100, color=ink)

        destination = args.output / f"network_kdv_readme_{theme}.png"
        fig.savefig(destination, dpi=200, transparent=True)
        plt.close(fig)
        outputs[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()

receipt = {
    "source_image": "docs/pages/examples/images/network_kdv.png",
    "source_sha256": source_sha256,
    "poisson_source_sha256": hashlib.sha256(args.poisson_source.read_bytes()).hexdigest(),
    "poisson_vertical_padding_pixels": padding,
    "source_crop_bounds_xywh": [[x, y, 718, 718] for x, y in source_bounds],
    "times": list(times), "temporal_interpolation": False,
    "display_width_pixels": width, "panel_width_pixels": side,
    "upper_figure_width_pixels": 332,
    "requested_upper_to_panel_ratio": 1.3 * 1.4,
    "labels": [f"t = {t}" for t in times] + ["u > 0", "u < 0", r"\mathrm{Width} \propto |u|"],
    "label_renderer": "Matplotlib MathText, Computer Modern",
    "outputs_sha256": outputs,
}
(args.output / "network_kdv_readme.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(f"README figures: {args.output}")
