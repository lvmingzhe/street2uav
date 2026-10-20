"""Model-free check of the FA-R-XL night pairing: do a label map's class boundaries lie on the image's edges?

For an image I and a label map L (both resized to 576 x 324): G = gradient magnitude of the grayscale image (central
differences), B = pixels where L changes class towards the right or bottom neighbour, dilated by one pixel. The
alignment score is mean(G on B) / mean(G): about 1 if the boundaries are unrelated to the image, clearly above 1 if
they follow its edges. Every night image is scored against all 50 night label maps; as a control, every clear (day)
image is scored against all 112 day label maps (the official split pairs day images by file name).
  python night_fix/scripts/edge_alignment_check.py --data_root DATA [--out night_fix/edge_alignment.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]  # night_fix/
SIZE = (576, 324)
FA_COLORS = np.array([[128, 0, 0], [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [64, 0, 128],
                      [64, 64, 0], [0, 0, 0]], dtype=np.uint8)


def gradient(path):
    g = np.asarray(Image.open(path).convert("L").resize(SIZE, Image.BILINEAR), dtype=np.float32)
    gx, gy = np.zeros_like(g), np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def boundaries(path):
    rgb = np.asarray(Image.open(path).convert("RGB").resize(SIZE, Image.NEAREST))
    lab = np.full(rgb.shape[:2], len(FA_COLORS), dtype=np.int16)
    for i, c in enumerate(FA_COLORS):
        lab[np.all(rgb == c, axis=2)] = i
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    d = b.copy()  # one-pixel dilation
    d[1:, :] |= b[:-1, :]
    d[:-1, :] |= b[1:, :]
    d[:, 1:] |= b[:, :-1]
    d[:, :-1] |= b[:, 1:]
    return d


def analyse(base, images, labels, partner):
    G = [gradient(base / "rgb" / x) for x in images]
    B = [boundaries(base / "gt" / y) for y in labels]
    S = np.array([[float(g[b].mean() / (g.mean() + 1e-6)) if b.any() else np.nan for b in B] for g in G])
    same = [labels.index(x) for x in images]  # images and labels share file names
    rank_same = [int((S[i] > S[i, same[i]]).sum()) + 1 for i in range(len(images))]
    out = {"n_images": len(images), "n_candidate_labels": len(labels),
           "same_name_is_best": int(sum(r == 1 for r in rank_same)),
           "same_name_rank_median": float(np.median(rank_same)), "same_name_rank_max": int(max(rank_same)),
           "score_same_name_mean": round(float(np.mean([S[i, same[i]] for i in range(len(images))])), 3),
           "score_all_pairs_mean": round(float(np.nanmean(S)), 3)}
    if partner is not None:
        idx = [labels.index(partner[x]) for x in images]
        rank_p = [int((S[i] > S[i, idx[i]]).sum()) + 1 for i in range(len(images))]
        out.update(official_partner_is_best=int(sum(r == 1 for r in rank_p)),
                   official_partner_rank_median=float(np.median(rank_p)),
                   score_official_partner_mean=round(float(np.mean([S[i, idx[i]] for i in range(len(images))])), 3),
                   same_name_beats_partner=int(sum(S[i, same[i]] > S[i, idx[i]] for i in range(len(images)))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default=str(HERE / "edge_alignment.json"))
    a = ap.parse_args()
    base = Path(a.data_root) / "FLYAWARE-R-XL"
    d = json.load(open(base / "splits/heterogeneous/test/test_user.json"))["user_data"]["test_user"]
    pairs = list(zip(d["x"], d["y"]))
    night_x = [x for x, _ in pairs if "/night/" in x]
    partner = {x: y for x, y in pairs if "/night/" in x}
    res = {"night": analyse(base, night_x, sorted(night_x), partner)}
    print("night", json.dumps(res["night"]), flush=True)
    day_x = [x for x, _ in pairs if "/day/" in x]
    res["day_control"] = analyse(base, day_x, sorted(day_x), None)
    print("day control", json.dumps(res["day_control"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
