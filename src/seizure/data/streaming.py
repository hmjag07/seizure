import sys
import time
import urllib.error
import urllib.request

import numpy as np


def get_range(url, start, end, retries=4):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # Dropped connections happen occasionally on long scans; back off and retry.
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def read_header(url):
    fixed = get_range(url, 0, 255)
    header_bytes = int(fixed[184:192])
    if fixed[192:197] == b"EDF+D":
        # Discontinuous files have gaps, so record i is not second i and read_seconds would misalign.
        raise ValueError(f"EDF+D (discontinuous) file not supported: {url}")
    n_records = int(fixed[236:244])
    record_dur = float(fixed[244:252])
    ns = int(fixed[252:256])
    sig = get_range(url, 256, header_bytes - 1)

    # Per-signal field widths, in EDF order.
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    names = ["label", "transducer", "dim", "pmin", "pmax", "dmin", "dmax", "prefilt", "nsamp", "reserved"]
    out, off = {}, 0
    for name, w in zip(names, widths):
        out[name] = [sig[off + i * w: off + (i + 1) * w].decode(errors="replace").strip() for i in range(ns)]
        off += w * ns
    nsamp = [int(x) for x in out["nsamp"]]
    pmin, pmax, dmin, dmax = (np.array(out[k], dtype=float) for k in ("pmin", "pmax", "dmin", "dmax"))
    # physical = pmin + (digital - dmin) * (pmax - pmin) / (dmax - dmin) = digital * gain + offset
    gain = (pmax - pmin) / (dmax - dmin)
    return dict(header_bytes=header_bytes, n_records=n_records, record_dur=record_dur,
                labels=out["label"], units=out["dim"], nsamp=nsamp, fs=[n / record_dur for n in nsamp],
                gain=gain, offset=pmin - dmin * gain, dmin=dmin, dmax=dmax)


def read_seconds(url, h, t0, t1):
    """Raw int16 samples per signal for the whole records covering [t0, t1) seconds."""
    rec_samples = sum(h["nsamp"])
    r0 = int(t0 // h["record_dur"])
    r1 = min(int(-(-t1 // h["record_dur"])), h["n_records"])
    data = get_range(url, h["header_bytes"] + r0 * 2 * rec_samples, h["header_bytes"] + r1 * 2 * rec_samples - 1)
    recs = np.frombuffer(data, dtype="<i2").reshape(r1 - r0, rec_samples)
    bounds = np.cumsum([0] + h["nsamp"])
    return [recs[:, a:b].ravel() for a, b in zip(bounds[:-1], bounds[1:])], len(data)


def read_physical(url, h, t0, t1):
    """Signals over exactly [t0, t1) seconds in physical units, skipping EDF Annotations.

    Returns {label: (samples, fs, unit)}. A repeated label (CHB-MIT lists T8-P8 twice)
    gets a suffix, e.g. "T8-P8 (2)", instead of overwriting the first.
    """
    raw, _ = read_seconds(url, h, t0, t1)
    start = int(t0 // h["record_dur"]) * h["record_dur"]
    out = {}
    for i, (label, fs, unit) in enumerate(zip(h["labels"], h["fs"], h["units"])):
        if label == "EDF Annotations":
            continue
        n = 2
        while label in out:
            label = f"{h['labels'][i]} ({n})"
            n += 1
        a, b = round((t0 - start) * fs), round((t1 - start) * fs)
        out[label] = (raw[i][a:b] * h["gain"][i] + h["offset"][i], fs, unit)
    return out


if __name__ == "__main__":
    for url in sys.argv[1:]:
        h = read_header(url)
        print(url.rsplit("/", 1)[-1])
        print("  records:", h["n_records"], "x", h["record_dur"], "s ->", round(h["n_records"] * h["record_dur"] / 3600, 2), "h")
        for label, (x, fs, unit) in list(read_physical(url, h, 1000, 1004).items())[:6]:
            print(f"  {label:16s} {fs:5.0f} Hz  n={len(x):4d}  range [{x.min():9.3f}, {x.max():9.3f}] {unit}")
