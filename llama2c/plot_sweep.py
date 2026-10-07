#!/usr/bin/env python3
"""Plot direct/graph results emitted by profile_cuda.sh sweep.

Two panels with independent y-scales (never a shared or dual axis):
GPU-time speedup on a linear scale and CPU-submission speedup on a log scale.
Each point is the mean of per-trial direct/graph ratios; the whisker spans the
minimum and maximum trial ratio.
"""
import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
AXIS = "#8a8984"
SERIES = "#2a78d6"


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def load(path):
    """Group valid rows by configuration, pairing direct and graph by trial."""
    groups = {}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("valid") != "1" or row.get("exit_code") != "0":
                continue
            key = (row["batch"], row["size"], row["rows"], row["repeats"], row["copies"],
                   row.get("matmul") or "naive")
            groups.setdefault(key, defaultdict(dict))[row["trial"]][row["mode"]] = row
    results = []
    for key, trials in groups.items():  # dicts keep sweep order
        pairs = [t for t in trials.values() if "direct" in t and "graph" in t]
        if not pairs:
            continue
        def values(mode, field):
            return [float(t[mode][field]) for t in pairs]
        def optional_mean(mode, field):
            values_ = [float(t[mode][field]) for t in pairs if t[mode].get(field)]
            return mean(values_) if values_ else ""
        def ratios(field):
            return [float(t["direct"][field]) / float(t["graph"][field]) for t in pairs]
        gpu = ratios("gpu_ms_per_iter")
        submit = ratios("submit_us_per_iter")
        setup = mean([sum(float(t["graph"][f]) for f in
                          ("graph_capture_ms", "graph_instantiate_ms", "graph_upload_ms"))
                      for t in pairs])
        wall_saving = mean(values("direct", "wall_ms_per_iter")) - mean(values("graph", "wall_ms_per_iter"))
        results.append({
            "batch": int(key[0]), "size": int(key[1]), "rows": int(key[2]),
            "repeats": int(key[3]), "copies": key[4], "matmul": key[5], "trials": len(pairs),
            "min_waves": optional_mean("direct", "min_waves"),
            "direct_gbps": optional_mean("direct", "achieved_gbps"),
            "graph_gbps": optional_mean("graph", "achieved_gbps"),
            "direct_gpu_ms": mean(values("direct", "gpu_ms_per_iter")),
            "graph_gpu_ms": mean(values("graph", "gpu_ms_per_iter")),
            "gpu_speedup": mean(gpu), "gpu_speedup_min": min(gpu), "gpu_speedup_max": max(gpu),
            "direct_submit_us": mean(values("direct", "submit_us_per_iter")),
            "graph_submit_us": mean(values("graph", "submit_us_per_iter")),
            "submit_speedup": mean(submit), "submit_speedup_min": min(submit),
            "submit_speedup_max": max(submit),
            "graph_setup_ms": setup,
            "break_even_replays": math.ceil(setup / wall_saving) if wall_saving > 0 else "",
        })
    return results


def linear_ticks(lo, hi, count=6):
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    ticks = []
    value = first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def log_ticks(lo, hi):
    ticks = []
    for exponent in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10 ** exponent
            if lo <= value <= hi:
                ticks.append(value)
    return ticks


def format_us(value_us):
    if value_us >= 1000:
        return f"{value_us / 1000:g} ms"
    return f"{value_us:g} µs"


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def config_label(result):
    copies = "copies" if result["copies"] == "copy" else "no copies"
    matmul = ", coalesced matmul" if result["matmul"] == "coalesced" else ""
    return (f'batch {result["batch"]}, size {result["size"]}, rows {result["rows"]}, '
            f'×{result["repeats"]}, {copies}{matmul}')


class LabelPlacer:
    """Greedy placement of short point labels that avoids earlier labels."""

    def __init__(self):
        self.boxes = []

    def place(self, x, y, text, bounds):
        width, height = 7 * len(text) + 2, 13
        candidates = [(x + 8, y - 8), (x + 8, y + 16), (x - 8 - width, y - 8),
                      (x - 8 - width, y + 16), (x - width / 2, y - 14), (x - width / 2, y + 22)]
        for cx, cy in candidates:
            box = (cx, cy - height + 3, cx + width, cy + 3)
            inside = bounds[0] <= box[0] and box[2] <= bounds[2] and bounds[1] <= box[1] and box[3] <= bounds[3]
            if inside and not any(self._overlaps(box, other) for other in self.boxes):
                self.boxes.append(box)
                return cx, cy
        self.boxes.append((candidates[0][0], candidates[0][1] - height + 3,
                           candidates[0][0] + width, candidates[0][1] + 3))
        return candidates[0]

    @staticmethod
    def _overlaps(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def svg_plot(results, output):
    width = 1200
    left, right, top, gap = 80, 30, 100, 100
    panel_w = (width - left - right - gap) / 2
    panel_h = 380
    key_columns = 2
    key_rows = math.ceil(len(results) / key_columns)
    key_top = top + panel_h + 70
    height = key_top + key_rows * 20 + 20

    x_values = [r["direct_gpu_ms"] * 1000 for r in results]
    x_lo = 10 ** math.floor(math.log10(min(x_values)))
    x_hi = 10 ** math.ceil(math.log10(max(x_values)))

    def xmap(value_us, x0):
        return x0 + (math.log10(value_us) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo)) * panel_w

    gpu_lo = min(0.9, min(r["gpu_speedup_min"] for r in results) - 0.05)
    gpu_hi = max(1.1, max(r["gpu_speedup_max"] for r in results) + 0.05)
    submit_lo = min(1.0, min(r["submit_speedup_min"] for r in results)) / 1.25
    submit_hi = max(r["submit_speedup_max"] for r in results) * 1.25
    panels = [
        {"x0": left, "field": "gpu_speedup", "title": "GPU time per iteration: direct ÷ graph",
         "scale": "linear", "lo": gpu_lo, "hi": gpu_hi, "ticks": linear_ticks(gpu_lo, gpu_hi),
         "fmt": lambda v: f"{v:.2f}×"},
        {"x0": left + panel_w + gap, "field": "submit_speedup",
         "title": "CPU submission per iteration: direct ÷ graph (log)",
         "scale": "log", "lo": submit_lo, "hi": submit_hi, "ticks": log_ticks(submit_lo, submit_hi),
         "fmt": lambda v: f"{v:g}×"},
    ]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">',
             f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
             f'<text x="{left}" y="30" font-size="18" font-weight="600" fill="{TEXT}">'
             'CUDA Graph speedup across the size sweep</text>',
             f'<text x="{left}" y="52" font-size="13" fill="{TEXT_SECONDARY}">'
             'Above 1× the graph is faster. Dot = mean of trials; whisker = min–max trial. '
             'Numbers refer to the configuration key below.</text>']
    for panel in panels:
        x0 = panel["x0"]
        if panel["scale"] == "linear":
            ymap = lambda v, p=panel: top + panel_h - (v - p["lo"]) / (p["hi"] - p["lo"]) * panel_h
        else:
            ymap = lambda v, p=panel: top + panel_h - (
                (math.log10(v) - math.log10(p["lo"])) / (math.log10(p["hi"]) - math.log10(p["lo"])) * panel_h)
        parts.append(f'<text x="{x0}" y="{top - 12}" font-size="14" font-weight="600" fill="{TEXT}">'
                     f'{escape(panel["title"])}</text>')
        for tick in panel["ticks"]:
            y = ymap(tick)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="{GRID}"/>')
            parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" '
                         f'fill="{TEXT_SECONDARY}">{panel["fmt"](tick)}</text>')
        parity = ymap(1.0)
        parts.append(f'<line x1="{x0}" y1="{parity:.1f}" x2="{x0 + panel_w}" y2="{parity:.1f}" '
                     f'stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x0 + 4}" y="{parity - 6:.1f}" font-size="12" '
                     f'fill="{TEXT_SECONDARY}">parity</text>')
        exponent = math.floor(math.log10(x_lo))
        while 10 ** exponent <= x_hi:
            x = xmap(10 ** exponent, x0)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}" stroke="{GRID}"/>')
            parts.append(f'<text x="{x:.1f}" y="{top + panel_h + 20}" text-anchor="middle" font-size="12" '
                         f'fill="{TEXT_SECONDARY}">{format_us(10 ** exponent)}</text>')
            exponent += 1
        parts.append(f'<line x1="{x0}" y1="{top + panel_h}" x2="{x0 + panel_w}" y2="{top + panel_h}" stroke="{AXIS}"/>')
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{top + panel_h + 44}" text-anchor="middle" font-size="13" '
                     f'fill="{TEXT_SECONDARY}">direct-mode GPU time per iteration (log)</text>')
        field = panel["field"]
        points = []
        for result in results:
            x = xmap(result["direct_gpu_ms"] * 1000, x0)
            points.append((x, ymap(result[field]), ymap(result[field + "_min"]), ymap(result[field + "_max"])))
        # Labels must avoid every mark (dots and whiskers), not only other labels.
        placer = LabelPlacer()
        placer.boxes = [(x - 6, min(y, y_lo, y_hi) - 6, x + 6, max(y, y_lo, y_hi) + 6)
                        for x, y, y_lo, y_hi in points]
        placer.boxes.append((x0, parity - 20, x0 + 50, parity))
        placer.boxes.append((x0, parity - 3, x0 + panel_w, parity + 3))
        marks, labels = [], []
        for index, (result, (x, y, y_min, y_max)) in enumerate(zip(results, points), start=1):
            tip = (f'{index}: {config_label(result)}&#10;{result[field]:.3f}× mean '
                   f'({result[field + "_min"]:.3f}–{result[field + "_max"]:.3f}, {result["trials"]} trials)')
            marks.append(f'<g><title>{escape(tip)}</title>'
                         f'<line x1="{x:.1f}" y1="{y_min:.1f}" x2="{x:.1f}" y2="{y_max:.1f}" '
                         f'stroke="{SERIES}" stroke-width="2" stroke-linecap="round"/>'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/>'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{SERIES}" stroke="{SURFACE}" '
                         'stroke-width="2"/></g>')
            lx, ly = placer.place(x, y, str(index), (x0, top, x0 + panel_w, top + panel_h))
            labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="12" fill="{TEXT_SECONDARY}">{index}</text>')
        parts.extend(marks + labels)
    column_w = (width - left - right) / key_columns
    parts.append(f'<text x="{left}" y="{key_top - 16}" font-size="13" font-weight="600" fill="{TEXT}">'
                 'Configurations</text>')
    for index, result in enumerate(results):
        column, row = divmod(index, key_rows)
        x = left + column * column_w
        y = key_top + row * 20
        parts.append(f'<text x="{x}" y="{y}" font-size="12" fill="{TEXT_SECONDARY}">'
                     f'<tspan font-weight="600" fill="{TEXT}">{index + 1}</tspan>  '
                     f'{escape(config_label(result))}  ·  GPU {result["gpu_speedup"]:.2f}×, '
                     f'submit {result["submit_speedup"]:.1f}×</text>')
    parts.append('</svg>')
    Path(output).write_text("\n".join(parts))


def write_png(svg):
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return None
    png = svg.with_suffix(".png")
    subprocess.run([converter, "--zoom=2", "-o", str(png), str(svg)], check=True)
    return png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    results = load(args.csv)
    if not results:
        raise SystemExit("no matched valid direct/graph rows found")
    output = args.output or args.csv.with_name(args.csv.stem + "-speedup.svg")
    svg_plot(results, output)
    summary = output.with_name(output.stem + ".csv")
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"plot={output}")
    png = write_png(output)
    if png:
        print(f"png={png}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
