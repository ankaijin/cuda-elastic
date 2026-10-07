#!/usr/bin/env python3
"""Timeline and kernel-duration plots from Nsight Systems SQLite exports.

Pass one export per mode (for example the direct and graph runs of
`profile_cuda.sh nsys-node`). Timed iterations are the `llama2c.iteration`
NVTX ranges; warm-up ranges are ignored. GPU work is tied to an iteration
through the correlation IDs of the CUDA runtime calls made inside its range,
which covers both direct launches and graph replays (every node of a replay
carries the correlation ID of its cudaGraphLaunch).

Outputs:
  <prefix>-timeline.svg   one representative iteration per mode, shared µs axis
  <prefix>-kernels.svg    per-kernel duration spread over all timed iterations

Node tracing adds overhead; use the unprofiled sweep numbers for speedups.
"""
import argparse
import shutil
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
AXIS = "#8a8984"
COPY = "#8a8984"
KERNELS = [  # fixed categorical order; color follows the kernel in every chart
    ("rmsnorm_kernel", "rmsnorm", "#2a78d6"),
    ("softmax_kernel", "softmax", "#eb6834"),
    ("matmul_kernel", "matmul", "#1baf7a"),
]
KERNEL_COLOR = {name: color for name, _, color in KERNELS}
KERNEL_LABEL = {name: label for name, label, _ in KERNELS}
ITERATION_RANGE = "llama2c.iteration"


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tables(db):
    return {row[0] for row in db.execute("select name from sqlite_master where type='table'")}


def load_trace(path):
    """Return (mode, iterations) where each iteration is a list of activities."""
    db = sqlite3.connect(path)
    present = tables(db)
    for required in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if required not in present:
            raise SystemExit(f"{path}: missing {required}; profile with "
                             "profile_cuda.sh nsys-node (--trace=cuda,nvtx --cuda-graph-trace=node)")
    strings = dict(db.execute("select id, value from StringIds"))
    ranges = [(start, end) for start, end, text, text_id in db.execute(
        "select start, end, text, textId from NVTX_EVENTS where end is not null order by start")
        if (text or strings.get(text_id)) == ITERATION_RANGE]
    if not ranges:
        raise SystemExit(f"{path}: no {ITERATION_RANGE} NVTX ranges found")
    calls = db.execute("select start, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
                       "order by start").fetchall()

    activities = defaultdict(list)
    for start, end, stream, correlation, name_id in db.execute(
            "select start, end, streamId, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        activities[correlation].append({"start": start, "end": end, "stream": stream,
                                        "name": strings.get(name_id, "kernel")})
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in present:
        for start, end, stream, correlation, size in db.execute(
                "select start, end, streamId, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
            activities[correlation].append({"start": start, "end": end, "stream": stream,
                                            "name": "copy", "bytes": size})
    db.close()

    iterations, graph_launches, call_index = [], 0, 0
    for range_start, range_end in ranges:
        while call_index < len(calls) and calls[call_index][0] < range_start:
            call_index += 1
        work = []
        index = call_index
        while index < len(calls) and calls[index][0] <= range_end:
            _, correlation, name_id = calls[index]
            if strings.get(name_id, "").startswith("cudaGraphLaunch"):
                graph_launches += 1
            work.extend(activities.get(correlation, ()))
            index += 1
        if work:
            iterations.append(sorted(work, key=lambda item: item["start"]))
    if not iterations:
        raise SystemExit(f"{path}: no GPU activity correlated with {ITERATION_RANGE} ranges")
    mode = "graph" if graph_launches else "direct"
    return mode, iterations


def span(iteration):
    return max(item["end"] for item in iteration) - min(item["start"] for item in iteration)


def representative(iterations, requested):
    if requested is not None:
        if not 0 <= requested < len(iterations):
            raise SystemExit(f"--iteration must be in [0, {len(iterations) - 1}]")
        return requested
    ordered = sorted(range(len(iterations)), key=lambda i: span(iterations[i]))
    return ordered[len(ordered) // 2]


def nice_step(extent, count):
    raw = extent / count
    magnitude = 10 ** len(str(int(raw))) / 10 if raw >= 1 else 10 ** -len(str(int(1 / raw)))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)


def lane_label(items):
    names = [item["name"] for item in items if item["name"] != "copy"]
    dominant = max(set(names), key=names.count) if names else "copy"
    return f'{KERNEL_LABEL.get(dominant, dominant)} stream'


def legend(parts, x, y, include_copy):
    entries = [(label, color) for _, label, color in KERNELS]
    if include_copy:
        entries.append(("device→host copy", COPY))
    for label, color in entries:
        parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{y}" font-size="13" fill="{TEXT_SECONDARY}">{escape(label)}</text>')
        x += 36 + 7.5 * len(label)


def timeline(traces, requested, output):
    width, left, right = 1200, 150, 40
    plot_w = width - left - right
    lane_h, panel_gap = 34, 62
    chosen = []
    for mode, iterations in traces:
        index = representative(iterations, requested)
        chosen.append((mode, index, len(iterations), iterations[index]))
    extent_us = max(span(items) for *_, items in chosen) / 1000
    step = nice_step(extent_us, 8)
    axis_end = step * -(-extent_us // step)
    has_copy = any(item["name"] == "copy" for *_, items in chosen for item in items)

    top = 100
    height = top + sum(len({i["stream"] for i in items}) * lane_h + panel_gap for *_, items in chosen) + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">',
             f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
             f'<text x="{left}" y="30" font-size="18" font-weight="600" fill="{TEXT}">'
             'One representative iteration per mode</text>',
             f'<text x="{left}" y="52" font-size="13" fill="{TEXT_SECONDARY}">'
             'Median-span timed iteration; time from its first GPU activity, shared scale. '
             'Node tracing adds overhead.</text>']
    legend(parts, left, 78, has_copy)
    y = top
    for mode, index, count, items in chosen:
        t0 = min(item["start"] for item in items)
        # Same lane order in every panel: by the kernel each stream carries.
        order = {f"{label} stream": rank for rank, (_, label, _) in enumerate(KERNELS)}
        streams = sorted({item["stream"] for item in items}, key=lambda s: order.get(
            lane_label([i for i in items if i["stream"] == s]), len(order)))
        panel_h = len(streams) * lane_h
        parts.append(f'<text x="{left}" y="{y + 12}" font-size="14" font-weight="600" fill="{TEXT}">'
                     f'{escape(mode)} · iteration {index + 1} of {count} · span {span(items) / 1000:.1f} µs</text>')
        body = y + 22
        tick = 0.0
        while tick <= axis_end + step * 1e-9:
            x = left + tick / axis_end * plot_w
            parts.append(f'<line x1="{x:.1f}" y1="{body}" x2="{x:.1f}" y2="{body + panel_h}" stroke="{GRID}"/>')
            parts.append(f'<text x="{x:.1f}" y="{body + panel_h + 16}" text-anchor="middle" font-size="12" '
                         f'fill="{TEXT_SECONDARY}">{tick:g} µs</text>')
            tick += step
        end_x = left + span(items) / 1000 / axis_end * plot_w
        parts.append(f'<line x1="{end_x:.1f}" y1="{body - 4}" x2="{end_x:.1f}" y2="{body + panel_h}" '
                     f'stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="4 3"/>')
        for lane, stream in enumerate(streams):
            lane_items = [item for item in items if item["stream"] == stream]
            lane_y = body + lane * lane_h
            parts.append(f'<text x="{left - 10}" y="{lane_y + lane_h / 2 + 4}" text-anchor="end" font-size="12" '
                         f'fill="{TEXT_SECONDARY}">{escape(lane_label(lane_items))}</text>')
            for item in lane_items:
                x = left + (item["start"] - t0) / 1000 / axis_end * plot_w
                w = (item["end"] - item["start"]) / 1000 / axis_end * plot_w
                # 1px surface gap on each side so back-to-back nodes stay distinct.
                if w > 4:
                    x, w = x + 1, w - 2
                w = max(w, 1.5)
                color = COPY if item["name"] == "copy" else KERNEL_COLOR.get(item["name"], AXIS)
                label = "device→host copy" if item["name"] == "copy" else item["name"]
                detail = f', {item["bytes"]} bytes' if "bytes" in item else ""
                tip = (f'{label}: {(item["end"] - item["start"]) / 1000:.2f} µs, starts at '
                       f'{(item["start"] - t0) / 1000:.2f} µs{detail}')
                parts.append(f'<rect x="{x:.1f}" y="{lane_y + 6}" width="{w:.1f}" height="{lane_h - 12}" rx="3" '
                             f'fill="{color}"><title>{escape(tip)}</title></rect>')
        parts.append(f'<line x1="{left}" y1="{body + panel_h}" x2="{left + plot_w}" y2="{body + panel_h}" '
                     f'stroke="{AXIS}"/>')
        y = body + panel_h + panel_gap - 22
    parts.append('</svg>')
    Path(output).write_text("\n".join(parts))


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def kernel_stats(traces):
    stats = []
    for name, label, color in KERNELS:
        for mode, iterations in traces:
            durations = [(item["end"] - item["start"]) / 1000 for items in iterations
                         for item in items if item["name"] == name]
            if durations:
                stats.append({"kernel": name, "label": label, "color": color, "mode": mode,
                              "n": len(durations), "min": min(durations), "max": max(durations),
                              "p10": percentile(durations, 0.1), "p90": percentile(durations, 0.9),
                              "median": statistics.median(durations)})
    return stats


def kernel_plot(stats, output):
    width, left, right, top, row_h, group_gap = 1200, 190, 150, 110, 30, 18
    plot_w = width - left - right
    groups = len({row["kernel"] for row in stats})
    height = top + len(stats) * row_h + groups * group_gap + 50
    extent = max(row["max"] for row in stats)
    step = nice_step(extent, 8)
    axis_end = step * -(-extent // step)
    xmap = lambda value: left + value / axis_end * plot_w
    bottom = height - 50

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">',
             f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
             f'<text x="{left}" y="30" font-size="18" font-weight="600" fill="{TEXT}">'
             'Kernel duration by launch mode</text>',
             f'<text x="{left}" y="52" font-size="13" fill="{TEXT_SECONDARY}">'
             'All timed iterations. Dot = median; bar = 10th–90th percentile; thin line = min–max.</text>',
             f'<text x="{left}" y="72" font-size="13" fill="{TEXT_SECONDARY}">'
             'Durations exclude gaps between kernels, so they cannot be summed into iteration time.</text>']
    tick = 0.0
    while tick <= axis_end + step * 1e-9:
        x = xmap(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="12" '
                     f'fill="{TEXT_SECONDARY}">{tick:g} µs</text>')
        tick += step
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left + plot_w}" y2="{bottom}" stroke="{AXIS}"/>')
    y = top
    previous = None
    for row in stats:
        if previous is not None and row["kernel"] != previous:
            y += group_gap
        previous = row["kernel"]
        mid = y + row_h / 2
        parts.append(f'<text x="{left - 12}" y="{mid + 4}" text-anchor="end" font-size="13" fill="{TEXT}">'
                     f'{escape(row["label"])} · <tspan fill="{TEXT_SECONDARY}">{escape(row["mode"])}</tspan></text>')
        tip = (f'{row["label"]} ({row["mode"]}): median {row["median"]:.2f} µs, p10–p90 '
               f'{row["p10"]:.2f}–{row["p90"]:.2f} µs, min–max {row["min"]:.2f}–{row["max"]:.2f} µs, '
               f'n={row["n"]}')
        parts.append(f'<g><title>{escape(tip)}</title>'
                     f'<rect x="{left}" y="{y}" width="{plot_w}" height="{row_h}" fill="transparent"/>'
                     f'<line x1="{xmap(row["min"]):.1f}" y1="{mid}" x2="{xmap(row["max"]):.1f}" y2="{mid}" '
                     f'stroke="{row["color"]}" stroke-width="1.5"/>'
                     f'<rect x="{xmap(row["p10"]):.1f}" y="{mid - 5}" '
                     f'width="{max(xmap(row["p90"]) - xmap(row["p10"]), 2):.1f}" height="10" rx="4" '
                     f'fill="{row["color"]}" fill-opacity="0.35"/>'
                     f'<circle cx="{xmap(row["median"]):.1f}" cy="{mid}" r="5" fill="{row["color"]}" '
                     f'stroke="{SURFACE}" stroke-width="2"/></g>')
        parts.append(f'<text x="{left + plot_w + 12}" y="{mid + 4}" font-size="12" fill="{TEXT_SECONDARY}">'
                     f'{row["median"]:.2f} µs (n={row["n"]})</text>')
        y += row_h
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite", type=Path, nargs="+", help="one Nsight Systems SQLite export per mode")
    parser.add_argument("-o", "--output-prefix", type=Path,
                        help="output path prefix (default: next to the first export)")
    parser.add_argument("--iteration", type=int,
                        help="zero-based timed iteration to draw (default: median span)")
    args = parser.parse_args()
    traces = [load_trace(path) for path in args.sqlite]
    traces.sort(key=lambda trace: trace[0] != "direct")  # direct first, then graph
    prefix = args.output_prefix or args.sqlite[0].with_suffix("")
    timeline_svg = Path(f"{prefix}-timeline.svg")
    kernels_svg = Path(f"{prefix}-kernels.svg")
    timeline(traces, args.iteration, timeline_svg)
    stats = kernel_stats(traces)
    kernel_plot(stats, kernels_svg)
    for mode, iterations in traces:
        spans = [span(items) / 1000 for items in iterations]
        print(f"{mode}: iterations={len(iterations)} span_median_us={statistics.median(spans):.2f} "
              f"span_min_us={min(spans):.2f} span_max_us={max(spans):.2f}")
    for row in stats:
        print(f'  {row["label"]:8s} {row["mode"]:6s} median_us={row["median"]:.2f} '
              f'p10_us={row["p10"]:.2f} p90_us={row["p90"]:.2f} n={row["n"]}')
    for svg in (timeline_svg, kernels_svg):
        png = write_png(svg)
        print(f"plot={svg}" + (f" png={png}" if png else ""))


if __name__ == "__main__":
    main()
