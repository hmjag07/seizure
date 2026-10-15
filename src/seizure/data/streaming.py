"""HTTP range-request EDF reader for SeizeIT2 and CHB-MIT (see docs/data-access.md).

Smoke test:
    python -m seizure.data.streaming <edf_url> [<edf_url> ...]
"""
import struct
import sys
import urllib.request


def get_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def read_header(url):
    fixed = get_range(url, 0, 255)
    header_bytes = int(fixed[184:192])
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
    return dict(header_bytes=header_bytes, n_records=n_records, record_dur=record_dur,
                labels=out["label"], nsamp=nsamp, fs=[n / record_dur for n in nsamp])


def read_seconds(url, h, t0, t1):
    """Return raw int16 samples per signal for records covering [t0, t1) seconds."""
    rec_bytes = 2 * sum(h["nsamp"])
    r0, r1 = int(t0 // h["record_dur"]), int(-(-t1 // h["record_dur"]))
    data = get_range(url, h["header_bytes"] + r0 * rec_bytes, h["header_bytes"] + r1 * rec_bytes - 1)
    per_sig = [[] for _ in h["nsamp"]]
    for r in range(r1 - r0):
        off = r * rec_bytes
        for i, n in enumerate(h["nsamp"]):
            per_sig[i].extend(struct.unpack_from(f"<{n}h", data, off))
            off += 2 * n
    return per_sig, len(data)


if __name__ == "__main__":
    for url in sys.argv[1:]:
        h = read_header(url)
        print(url.rsplit("/", 1)[-1])
        print("  records:", h["n_records"], "x", h["record_dur"], "s ->", round(h["n_records"] * h["record_dur"] / 3600, 2), "h")
        print("  signals:", list(zip(h["labels"], h["fs"]))[:6], "..." if len(h["labels"]) > 6 else "")
        sig, nbytes = read_seconds(url, h, 1000, 1004)
        print("  fetched 4 s =", nbytes, "bytes; samples/signal:", [len(s) for s in sig][:6], "first values:", sig[0][:5])
