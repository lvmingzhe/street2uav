"""Rebuild the corrected FA-R-XL test split from the official one (CPU only, no model involved).

The official split file pairs each of the 50 night test images with the label map of another night image (the label
list is the sorted image list rotated by 35 positions); the clear, fog and rain images are paired by file name. The
corrected split pairs every night image with the label map of the same file name and is otherwise identical.

Checks: the official file has the published SHA-256; image lists are equal; exactly the 50 night pairs change; every
corrected pair is same-name; the rebuilt file is byte-identical to night_fix/test_user_nightfix.json.
  python night_fix/scripts/make_corrected_split.py --data_root DATA [--out_dir night_fix]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # night_fix/
OFFICIAL_SHA256 = "736ad78ea00f6d145acb797c48ac84b56e3ffd7186c63028ea077fc741050742"
CORRECTED_SHA256 = "21c030066436c053b3fbacbee9f468d7bae7d2946632352402eb41732f7987de"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=str(HERE))
    a = ap.parse_args()
    official = Path(a.data_root) / "FLYAWARE-R-XL/splits/heterogeneous/test/test_user.json"
    raw = official.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    print(f"official split {official}: sha256 {sha} ({'as published' if sha == OFFICIAL_SHA256 else 'DIFFERENT'})")
    split = json.loads(raw)
    u = split["user_data"]["test_user"]
    xs, ys = u["x"], list(u["y"])
    night = [i for i, x in enumerate(xs) if "/night/" in x]
    mismatched = [i for i, (x, y) in enumerate(zip(xs, ys)) if x != y]
    assert mismatched == night and len(night) == 50, "expected exactly the 50 night pairs to differ"
    sorted_night = sorted(xs[i] for i in night)
    shift = next(k for k in range(50) if [ys[i] for i in night] == sorted_night[k:] + sorted_night[:k])
    print(f"{len(xs)} test pairs; the night labels are the sorted night images rotated by {shift} positions")

    rows = [{"image": xs[i], "official_label": ys[i], "corrected_label": xs[i],
             "index_offset": int(Path(ys[i]).stem) - int(Path(xs[i]).stem)} for i in night]
    u["y"] = [x if "/night/" in x else y for x, y in zip(xs, ys)]
    assert all(x == y for x, y in zip(u["x"], u["y"]))
    text = json.dumps(split)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_user_nightfix.json").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"corrected split: sha256 {sha} ({'as published' if sha == CORRECTED_SHA256 else 'DIFFERENT'})")
    with open(out / "night_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'test_user_nightfix.json'} and {out / 'night_pairs.csv'}")


if __name__ == "__main__":
    main()
