#!/usr/bin/env python3
"""Compare llama2c output dumps (--dump=FILE, layout in llama2c_common.h).

Usage: compare_dumps.py REFERENCE.bin OTHER.bin [OTHER.bin ...]

Every OTHER dump must come from the same inputs (same header) and match the
REFERENCE dump within the oracle tolerances of llama2c_common.h:
  normalized  |a - b| <= 3e-4
  softmax     |a - b| <= 1e-9 + 1e-3 |reference|
  product     |a - b| <= product_tolerance[row] (the tolerance the oracle
              applied; for random inputs the fp32 dot-product bound)
It also reports, for every dump, the largest product error against the
oracle's double reference as a fraction of the tolerance. Uses numpy when
available, else plain Python (slow at the sheet shape). Exits 1 on mismatch.
"""
import os
import struct
import sys
from array import array

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback path
    np = None

RMS_TOLERANCE = 3.0e-4
SOFTMAX_RELATIVE = 1.0e-3
SOFTMAX_ABSOLUTE = 1.0e-9


def load(path):
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != b"L2CDUMP1":
        raise SystemExit(f"{path}: not a llama2c dump")
    batch, size, rows, inputs = struct.unpack_from("<4i", data, 8)
    (seed,) = struct.unpack_from("<Q", data, 24)
    n = batch * size
    offset = 32
    sections = {}
    for name, count, code in (("normalized", n, "f"), ("softmax", n, "f"), ("product", rows, "f"),
                              ("product_tolerance", rows, "f"), ("product_reference", rows, "d")):
        width = 4 if code == "f" else 8
        chunk = data[offset:offset + count * width]
        if len(chunk) != count * width:
            raise SystemExit(f"{path}: truncated in {name}")
        if np is not None:
            sections[name] = np.frombuffer(chunk, dtype="<f4" if code == "f" else "<f8")
        else:
            values = array(code)
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            sections[name] = values
        offset += count * width
    if offset != len(data):
        raise SystemExit(f"{path}: {len(data) - offset} trailing bytes")
    header = dict(batch=batch, size=size, rows=rows, inputs="random" if inputs else "deterministic", seed=seed)
    return header, sections


def max_violation(actual, expected, tolerance_of):
    """Largest |a - e| / tolerance and largest |a - e| (plain Python path)."""
    worst_ratio = worst_abs = 0.0
    for index, (a, e) in enumerate(zip(actual, expected)):
        diff = abs(a - e)
        if diff != diff:  # NaN
            return float("inf"), float("inf")
        worst_abs = max(worst_abs, diff)
        worst_ratio = max(worst_ratio, diff / tolerance_of(index, e))
    return worst_ratio, worst_abs


def compare(ref, other):
    """Returns {output: (max |diff| / tolerance, max |diff|)}."""
    out = {}
    if np is not None:
        def stat(a, b, tol):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if not np.all(np.isfinite(diff)):
                return float("inf"), float("inf")
            return float(np.max(diff / tol)) if diff.size else 0.0, float(np.max(diff)) if diff.size else 0.0
        out["normalized"] = stat(other["normalized"], ref["normalized"], RMS_TOLERANCE)
        out["softmax"] = stat(other["softmax"], ref["softmax"],
                              SOFTMAX_ABSOLUTE + SOFTMAX_RELATIVE * np.abs(ref["softmax"].astype(np.float64)))
        out["product"] = stat(other["product"], ref["product"], ref["product_tolerance"].astype(np.float64))
    else:
        out["normalized"] = max_violation(other["normalized"], ref["normalized"], lambda i, e: RMS_TOLERANCE)
        out["softmax"] = max_violation(other["softmax"], ref["softmax"],
                                       lambda i, e: SOFTMAX_ABSOLUTE + SOFTMAX_RELATIVE * abs(e))
        tol = ref["product_tolerance"]
        out["product"] = max_violation(other["product"], ref["product"], lambda i, e: tol[i])
    return out


def product_error(sections):
    """Max |product - double reference| / tolerance and max |product - reference|."""
    if np is not None:
        diff = np.abs(sections["product"].astype(np.float64) - sections["product_reference"])
        return float(np.max(diff / sections["product_tolerance"])), float(np.max(diff))
    tol = sections["product_tolerance"]
    return max_violation(sections["product"], sections["product_reference"], lambda i, e: tol[i])


def identical(a, b):
    if np is not None:
        return all(np.array_equal(a[k].view(np.uint8), b[k].view(np.uint8)) for k in ("normalized", "softmax", "product"))
    return all(a[k].tobytes() == b[k].tobytes() for k in ("normalized", "softmax", "product"))


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    ref_header, ref = load(argv[1])
    ok = True
    print(f"  shape batch={ref_header['batch']} size={ref_header['size']} rows={ref_header['rows']} "
          f"inputs={ref_header['inputs']} seed={ref_header['seed']} ({'numpy' if np is not None else 'plain python'})")
    ratio, abs_err = product_error(ref)
    print(f"  {os.path.basename(argv[1])}: reference; product error vs double reference {abs_err:.3e} ({ratio:.2e} of tolerance)")
    for path in argv[2:]:
        header, other = load(path)
        if header != ref_header:
            print(f"  {os.path.basename(path)}: header {header} != {ref_header}")
            ok = False
            continue
        stats = compare(ref, other)
        ratio, abs_err = product_error(other)
        same = identical(ref, other)
        passed = all(r <= 1.0 for r, _ in stats.values()) and ratio <= 1.0
        ok = ok and passed
        details = " ".join(f"{k}: max|diff|={a:.3e} ({r:.2e} of tol)" for k, (r, a) in stats.items())
        print(f"  {os.path.basename(path)}: {'PASS' if passed else 'FAIL'}{' (bitwise identical)' if same else ''}; {details}; "
              f"product error vs double reference {abs_err:.3e} ({ratio:.2e} of tolerance)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
