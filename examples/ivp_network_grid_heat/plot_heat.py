"""Render saved heat fields with a colour bar labelled u on the right."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
from scipy.interpolate import BarycentricInterpolator


def render(output, render_output=None):
    output = Path(output)
    render_output = output if render_output is None else Path(render_output)
    render_output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((output / "run.json").read_text())
    xy = np.asarray(metadata["vertices"])
    endpoints = np.asarray(metadata["endpoints"])
    source = output / "snapshots/snapshots_s1.h5"
    with h5py.File(source, "r") as file:
        task = file["tasks/u"]
        values = task[:]
        vertices = file["tasks/U"][:, :, 0]
        s = task.dims[2][0][:]
        times = file["scales/sim_time"][:]

    # Evaluate each saved spectral polynomial along its rod. Do not interpolate
    # in time, change the data, or adjust the colour scale between frames.
    uniform = np.linspace(0, 1, 33)
    interpolation = BarycentricInterpolator(s, np.eye(len(s)))
    temperature = values @ interpolation(uniform).T
    tail, head = xy[endpoints[:, 0]], xy[endpoints[:, 1]]
    points = tail[:, None, :] + uniform[None, :, None] * (head - tail)[:, None, :]
    segments = np.stack((points[:, :-1], points[:, 1:]), axis=2).reshape(-1, 2, 2)
    colours = (temperature[:, :, :-1] + temperature[:, :, 1:]) / 2

    background = "#faf9f6"
    cmap = matplotlib.colormaps["inferno"]
    norm = LogNorm(vmin=0.001, vmax=1, clip=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12}):
        fig = plt.figure(figsize=(10, 8), dpi=100, facecolor=background)
        ax = fig.add_axes((0.02, 0.04, 0.76, 0.90))
        centre = (xy.max(axis=0) + xy.min(axis=0)) / 2
        radius = 0.54 * np.max(np.ptp(xy, axis=0))
        ax.set(xlim=(centre[0] - radius, centre[0] + radius),
               ylim=(centre[1] - radius, centre[1] + radius), aspect="equal")
        ax.set_axis_off()
        linewidth = 2.4 if len(xy) > 50 else 8
        rods = LineCollection(
            segments, cmap=cmap, norm=norm, linewidth=linewidth, capstyle="round",
            antialiased=True, zorder=2,
        )
        ax.add_collection(rods)
        nodes = ax.scatter(
            xy[:, 0], xy[:, 1], c=vertices[0], cmap=cmap, norm=norm,
            s=linewidth**2, linewidths=0, zorder=3,
        )

        # Match the colour bar to the network's height in data coordinates.
        span = np.max(np.ptp(xy, axis=0))
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
        colorbar_axis = ax.inset_axes(
            (xy[:, 0].max() + 0.12 * span, ymin, 0.035 * span, ymax - ymin),
            transform=ax.transData,
        )
        colorbar = fig.colorbar(
            rods, cax=colorbar_axis, ticks=(0.001, 0.01, 0.1, 1),
        )
        colorbar.ax.set_yticklabels(("0.001", "0.01", "0.1", "1"))
        colorbar.minorticks_off()
        colorbar.ax.set_title("$u$", fontsize=20, math_fontfamily="cm", pad=8)
        colorbar.ax.tick_params(labelsize=12, length=3, width=0.8, pad=6)
        colorbar.outline.set_linewidth(0.8)

        # One shared palette prevents GIF quantization from flickering. Keep
        # both the thermal colours and neutral background antialiasing.
        thermal = np.rint(255 * cmap(np.linspace(0, 1, 192))[:, :3]).astype(np.uint8)
        neutrals = np.rint(np.linspace([22, 24, 30], [250, 249, 246], 64)).astype(np.uint8)
        palette = Image.new("P", (1, 1))
        palette.putpalette(np.vstack((thermal, neutrals)).ravel().tolist())
        frames = []
        preview_indices = {
            0: "initial", int(np.argmin(np.abs(times - times[-1] * 0.2))): "spreading",
            len(times) - 1: "final",
        }
        # Redraw only the changing rods and junctions.
        for artist in (rods, nodes):
            artist.set_animated(True)
        fig.canvas.draw()
        background_pixels = fig.canvas.copy_from_bbox(fig.bbox)
        for index in range(len(times)):
            rods.set_array(colours[index].ravel())
            nodes.set_array(vertices[index])
            fig.canvas.restore_region(background_pixels)
            ax.draw_artist(rods)
            ax.draw_artist(nodes)
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            if index in preview_indices:
                rgb.save(render_output / f"heat_{preview_indices[index]}.png")
            frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        plt.close(fig)

    durations = [80] * len(frames)
    durations[0], durations[-1] = 800, 1200
    name = "network_disc_heat" if metadata.get("geometry") == "disc" else "network_grid_heat"
    animation = render_output / f"{name}.gif"
    frames[0].save(
        animation, save_all=True, append_images=frames[1:], duration=durations,
        loop=0, optimize=False, disposal=1,
    )
    with Image.open(animation) as saved:
        if saved.n_frames != len(times):
            raise RuntimeError("Animation is missing saved simulation states")
        for frame in range(saved.n_frames):
            saved.seek(frame)
            saved.load()
        dimensions = list(saved.size)
    receipt = {
        "source": str(source.resolve()), "animation": str(animation.resolve()),
        "frames": len(times), "dimensions": dimensions,
        "simulation_times": times.tolist(), "durations_ms": durations,
        "colour_map": "inferno", "colour_normalization": "fixed logarithmic, 0.001 to 1",
        "colour_floor": "temperatures at or below 0.001 use the lowest colour",
        "spatial_interpolation": "saved spectral polynomials, 33 points per rod",
        "temporal_interpolation": False,
        "annotations": True,
        "colour_bar": "right, with u above; same height as the network",
        "title": False, "time_annotation": False,
    }
    (render_output / "animation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Animation: {animation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("network_heat"))
    parser.add_argument("--render-output", type=Path, help="Save images separately from the simulation")
    args = parser.parse_args()
    render(args.output, args.render_output)
