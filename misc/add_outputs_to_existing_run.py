#!/usr/bin/env python
"""
Add the new GIFTwrap deliverables (.h5ad + flat_counts.N.tsv.gz) to an EXISTING output
directory, without rerunning the pipeline. Reuses the already-generated intermediates
(counts.N.h5 / counts.N.filtered.h5, probe_reads.tsv.gz, barcodes.tsv.gz, manifest.tsv).

Usage:
    python add_giftwrap_outputs.py <analysis_results_dir> [--overwrite]

Points at the edited giftwrap source tree so it uses the new write_gapfill_h5ad() and the
fixed compile_flatfile() (not the older installed package).
"""
import sys, argparse
from pathlib import Path

GIFTWRAP_SRC = "/data1/collab002/sail/projects/ongoing/tools/sail_giftwrap/sail_giftwrap/src"
sys.path.insert(0, GIFTWRAP_SRC)  # ensure the edited source wins over any installed giftwrap

from giftwrap.utils import read_h5_file, write_gapfill_h5ad, read_manifest, read_barcodes, compile_flatfile


def find_output_dirs(root: Path):
    """Any directory containing at least one counts.*.h5 is an output dir to process."""
    dirs = set()
    for h5 in root.rglob("counts.*.h5"):
        dirs.add(h5.parent)
    return sorted(dirs)


def process_dir(d: Path, overwrite: bool):
    manifest = read_manifest(d)
    barcodes = read_barcodes(d)
    barcode_list = barcodes.barcode.values.tolist()

    # 1) .h5ad for every counts*.h5 (mirror the name, swap extension)
    for h5 in sorted(d.glob("counts.*.h5")):
        out = h5.with_suffix(".h5ad")            # counts.1.h5 -> counts.1.h5ad ; counts.1.filtered.h5 -> counts.1.filtered.h5ad
        if out.exists() and not overwrite:
            print(f"  skip (exists): {out.name}")
        else:
            adata = read_h5_file(h5)
            write_gapfill_h5ad(adata, out)
            print(f"  wrote {out.name}  ({adata.shape[0]} cells x {adata.shape[1]} gapfills, {int(adata.X.sum())} UMIs)")

    # 2) flat_counts.N.tsv.gz per plex (one plex per dir here). Derive plex from the
    #    unfiltered counts filename: counts.<N>.h5
    plexes = sorted({h5.name.split(".")[1] for h5 in d.glob("counts.*.h5")})
    pr = d / "probe_reads.tsv.gz"
    if not pr.exists():
        print(f"  WARN: no probe_reads.tsv.gz in {d}, skipping flat_counts")
        return
    for plex in plexes:
        out = d / f"flat_counts.{plex}.tsv.gz"
        if out.exists() and not overwrite:
            print(f"  skip (exists): {out.name}")
            continue
        compile_flatfile(manifest, str(pr), barcode_list, plex, str(out))
        # quick sanity: count rows
        import gzip
        with gzip.open(out, "rt") as f:
            n = sum(1 for _ in f) - 1
        print(f"  wrote {out.name}  ({n} UMI rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dirs = find_output_dirs(args.root)
    print(f"Found {len(out_dirs)} output dir(s) with counts files.")
    for d in out_dirs:
        print(f"[{d}]")
        process_dir(d, args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
