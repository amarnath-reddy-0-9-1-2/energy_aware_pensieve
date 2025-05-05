#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
convert_traces.py

  - Reads all files in --indir
  - If a file has exactly one number per non-blank line → converts it
  - Otherwise assumes it’s already two-column and just copies it
  - Writes everything as '*.log' in --outdir

Usage:
  python2 convert_traces.py \
    --indir   /path/to/raw_traces \
    --outdir  /path/to/pensieve/sim/cooked_traces \
    [--interval-ms 1.0] [--packet-size 1500]
"""

import os
import shutil
import argparse

def is_one_column(path):
    """Return True if the first non-empty, non-comment line splits into exactly 1 token."""
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            return len(s.split()) == 1
    return False

def convert_file(inpath, outpath, interval_ms, packet_size):
    """Convert one-column Mahimahi → two-column Pensieve format."""
    with open(inpath, 'r') as fin:
        with open(outpath, 'w') as fout:
            for idx, line in enumerate(fin):
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                if len(parts) != 1:
                    continue
                try:
                    pkts_per_ms = float(parts[0])
                except ValueError:
                    continue

                # timestamp (s)
                time_s = idx * (interval_ms / 1000.0)
                # throughput (Mbps):
                # pkts/ms × packet_size (B) → bytes/ms
                # bytes/ms × 8 (bits/byte) × 1000 (ms→s) → bits/s
                # ÷1e6 → Mbps
                bw_mbps = pkts_per_ms * packet_size * 8 * (1000.0 / 1e6)

                fout.write("%.3f %.3f\n" % (time_s, bw_mbps))

def main():
    p = argparse.ArgumentParser(description="Mahimahi → Pensieve trace converter (Python 2)")
    p.add_argument("--indir",     required=True, help="Folder of raw .down/.up/.log/etc. files")
    p.add_argument("--outdir",    required=True, help="Where to deposit *.log two-column traces")
    p.add_argument("--interval-ms", type=float, default=1.0,
                   help="Sampling interval in ms (default: 1.0)")
    p.add_argument("--packet-size", type=int, default=1500,
                   help="Packet size in bytes (default: 1500)")
    args = p.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    for fname in os.listdir(args.indir):
        inpath = os.path.join(args.indir, fname)
        if not os.path.isfile(inpath):
            continue

        base, _ = os.path.splitext(fname)
        outname  = base + ".log"
        outpath  = os.path.join(args.outdir, outname)

        if is_one_column(inpath):
            print "Converting %-20s → %s" % (fname, outname)
            convert_file(inpath, outpath, args.interval_ms, args.packet_size)
        else:
            print "Copying   %-20s → %s" % (fname, outname)
            shutil.copyfile(inpath, outpath)

    print "All done! Two-column traces are in:", args.outdir

if __name__ == "__main__":
    main()

