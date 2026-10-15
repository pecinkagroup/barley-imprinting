"""
01_extract_thickness_profiles.py
================================

Preprocessing pipeline: per-seed rotation photos → thickness-along-length
profile in physical units (mm).

INPUT
-----
A directory containing one .zip per seed (or an unpacked folder per seed).
Each seed bundle must contain:
    * 35 rotation photos named "frame00.jpg" ... "frame34.jpg" (or similar
      frameNN.jpg naming, 10° between frames)
    * one CSV with one row per photo, columns including "Length" and "Width"
      in mm as reported by the scanner

The seed bundle name encodes genotype:
    * "Riso" anywhere in the filename  → genotype = "lys3.c"
    * "WT"   anywhere in the filename  → genotype = "WT"

OUTPUT
------
Written next to the input directory:
    thickness_profiles_per_seed.csv
        One row per successfully processed seed. Columns:
            seed          identifier (bundle basename)
            genotype      "WT" or "lys3.c"
            length_mm     seed length along major axis (mm)
            tilt_deg      in-plane principal-axis angle of the lateral
                          silhouette, before rotation correction (deg)
            t_000 ... t_099
                          thickness (mm) at 100 normalized positions along
                          the major axis. Position i corresponds to
                          normalized length (i + 0.5) / 100. Seeds are
                          oriented so the thicker half is at position 0.

PIPELINE (applied to every seed)
--------------------------------
1. Read all frames and the scanner CSV.
2. Pick the lateral frame = frame with maximum scanner Length (the seed
   lying broadside to the camera; this is the only view that resolves the
   full silhouette length AND thickness simultaneously).
3. Segment that frame:
       * foreground = gray > 28
       * intersect with "red-ish" mask (R > B) to remove the purple rod
       * keep the largest connected component
       * morphological close (5×5) then open (3×3) to tidy edges
4. Compute the mask's principal-axis angle from second-order image
   moments:
       θ = 0.5 * atan2(2·µ11, µ20 − µ02)
   and rotate the mask around its centroid by −θ so the major axis becomes
   horizontal. This removes side-view rotation, which would otherwise
   inflate measured thickness.
5. From the rotated mask, extract thickness(x) = vertical extent of the
   silhouette at each column x across the mask's horizontal extent.
6. Convert to mm using the ratio of scanner Length to length-in-pixels of
   the rotated silhouette.
7. Resample to 100 positions via linear interpolation.
8. Orient so the thicker half is at index 0: if mean(t[-n/3:]) >
   mean(t[:n/3]), reverse the profile.

DEPENDENCIES
------------
    numpy pandas scipy opencv-python
Install with:
    pip install numpy pandas scipy opencv-python
"""

import os, glob, zipfile, re, argparse, sys
import numpy as np
import pandas as pd
import cv2


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
N_PROF = 100            # number of positions along length axis
MIN_LEN_PX = 20         # reject silhouettes shorter than this


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def parse_name(name):
    """Infer genotype from a seed bundle filename."""
    if re.search(r"Riso", name, re.I):
        return "lys3.c"
    if "WT" in name:
        return "WT"
    return "Unknown"


def iter_seeds(data_dir):
    """Yield (seed_name, zip_path_or_None, folder_or_None) for every seed
    bundle in data_dir. Prefers .zip when both exist for the same seed."""
    seen = set()
    for zp in sorted(glob.glob(os.path.join(data_dir, "*.zip"))):
        seed = os.path.splitext(os.path.basename(zp))[0]
        seen.add(seed)
        yield seed, zp, None
    for folder in sorted(glob.glob(os.path.join(data_dir, "BARLEY*"))):
        if not os.path.isdir(folder):
            continue
        seed = os.path.basename(folder)
        if seed in seen:
            continue
        yield seed, None, folder


def load_frames_and_csv(zp, folder):
    """Return (dict{frame_idx: BGR image}, scanner_dataframe)."""
    imgs, df = {}, None
    if zp:
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            csvn = next(n for n in names if n.endswith(".csv"))
            with zf.open(csvn) as fh:
                df = pd.read_csv(fh)
            for n in names:
                m = re.search(r"frame(\d+)\.jpg$", n, flags=re.I)
                if m:
                    arr = np.frombuffer(zf.read(n), np.uint8)
                    imgs[int(m.group(1))] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        csvn = glob.glob(os.path.join(folder, "*.csv"))[0]
        df = pd.read_csv(csvn)
        for p in glob.glob(os.path.join(folder, "frame*.jpg")):
            m = re.search(r"frame(\d+)\.jpg$", os.path.basename(p), flags=re.I)
            if m:
                imgs[int(m.group(1))] = cv2.imread(p)
    return imgs, df


def segment(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (0,0), 1.0) # gausian blur to reduce noise and fragmentation works better than median blur
    fg = gray > 28 # keep at least 15 (lower value = darker, and gives more fragmentation and false positive pixels)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R = rgb[:, :, 0].astype(int); B = rgb[:, :, 2].astype(int)
    rod = (B > R + 10) # to exclude the rod which is blue and has a higher B value than R value (loosening criteria by adding 5 to R value works better than strict R > B)
    # rod = cv2.dilate(rod.astype(np.uint8), np.ones((5,5),np.uint8)).astype(bool) # dilate to remove small holes in the rod mask
    m = fg & (R + 10 > B) & (~rod) # to exclude the rod which is blue and has a higher B value than R value (loosening criteria by adding 5 to R value works better than strict R > B)
    num, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if num > 1:
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lab == keep)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((12,12),np.uint8)).astype(bool) # morphological close kernel to reduce fragmentation and false positive pixels (at least 10x10 works well)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,  np.ones((10,10),np.uint8)).astype(bool)
    return m


def principal_axis_angle(mask):
    """Angle (deg, counter-clockwise) of the major axis of a boolean mask
    w.r.t. horizontal, computed from image moments."""
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return 0.0
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    return -float(np.degrees(theta))   # flip sign because image y grows down


def rotate_mask(mask, angle_deg):
    """Rotate a boolean mask around its centroid so that -angle_deg is
    applied (use with angle from principal_axis_angle() to horizontalise)."""
    h, w = mask.shape
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return mask
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    R = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    rot = cv2.warpAffine(mask.astype(np.uint8), R, (w, h),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    return rot.astype(bool)

def rotate_gray(gray, mask, angle_deg):
    """Rotate a grayscale image around the mask centroid to match rotate_mask()."""
    h, w = gray.shape
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return gray
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    R = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    return cv2.warpAffine(gray, R, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=0)

def thickness_profile(mask, mm_per_px, n=N_PROF):
    """Thickness (mm) along the major axis at n normalized positions.
    Returns (profile_mm, length_mm) or (None, None) if mask is too small."""
    any_col = mask.any(axis=0)
    if not any_col.any():
        return None, None
    xs = np.where(any_col)[0]
    x0, x1 = xs.min(), xs.max()
    length_px = x1 - x0 + 1
    if length_px < MIN_LEN_PX:
        return None, None
    cols = np.arange(x0, x1 + 1)
    t_px = np.zeros_like(cols, dtype=float)
    for k, x in enumerate(cols):
        r = np.where(mask[:, x])[0]
        if len(r):
            t_px[k] = r.max() - r.min() + 1
    x_src = np.linspace(0, 1, length_px)
    x_tgt = (np.arange(n) + 0.5) / n   # bin midpoints 0.005 ... 0.995
    return np.interp(x_tgt, x_src, t_px) * mm_per_px, length_px * mm_per_px

def trim_narrow_rows(mask, min_width_px):
    """Trim mask rows that are narrower than `min_width_px`. Rod stubs
    hang above the seed body as thin vertical bars — each of their rows
    spans only ~5–15 pixels horizontally, while any row inside the seed
    body spans much more. Any row too narrow is discarded entirely.

    Absolute-pixel counterpart to trim_thin_tails, but operating on ROWS
    (horizontal width) instead of columns (vertical thickness).
    """
    out = mask.copy()
    for y in np.where(out.any(axis=1))[0]:
        xs = np.where(out[y, :])[0]
        width = xs.max() - xs.min() + 1
        if width < min_width_px:
            out[y, :] = False
    return out

def trim_thin_tails(mask, min_frac=0.2):
    """Trim thin end regions (awn remnants) from the mask. Any end column
    with thickness less than `min_frac` × the seed's max thickness is
    considered awn/debris and excluded."""
    xs = np.where(mask.any(axis=0))[0]
    if len(xs) < 3: return mask
    x0, x1 = xs.min(), xs.max()
    t = np.array([np.count_nonzero(mask[:, x]) for x in range(x0, x1+1)])
    thresh = t.max() * min_frac
    # first & last column above threshold
    keep = np.where(t >= thresh)[0]
    if len(keep) < 3: return mask
    left, right = x0 + keep.min(), x0 + keep.max()
    out = mask.copy()
    out[:, :left]     = False
    out[:, right+1:]  = False
    return out

def orient_embryo_left(prof, mask_rot, gray_rot, end_frac=0.35, erode_frac=0.04):
    # NEW: peel border pixels so we measure internal texture, not the edge
    if erode_frac > 0:
        seed_h = np.where(mask_rot.any(axis=1))[0]
        k = max(1, int((seed_h.max() - seed_h.min() + 1) * erode_frac))
        mask_use = cv2.erode(mask_rot.astype(np.uint8),
                             np.ones((k*2+1, k*2+1), np.uint8)).astype(bool)
        if not mask_use.any():
            mask_use = mask_rot   # fallback if erosion killed everything
    else:
        mask_use = mask_rot

    xs = np.where(mask_use.any(axis=0))[0]
    x0, x1 = xs.min(), xs.max()
    length = x1 - x0 + 1
    end_w = int(length * end_frac)
    gx = cv2.Sobel(gray_rot.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_rot.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    left_mask  = mask_use.copy(); left_mask [:, x0 + end_w :]      = False
    right_mask = mask_use.copy(); right_mask[:, : x1 - end_w + 1]  = False
    L = grad[left_mask ].mean() if left_mask.any()  else 0
    R = grad[right_mask].mean() if right_mask.any() else 0

    # Embryo pole = smoother end (smaller gradient).
    # If left is more textured, embryo is currently on the right → flip.
    return prof[::-1] if L > R else prof

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(data_dir, out_path):
    rows = []
    all_seeds = list(iter_seeds(data_dir))
    if not all_seeds:
        sys.exit(f"No seed bundles found in {data_dir}")
    print(f"Processing {len(all_seeds)} seed bundles from {data_dir} ...")

    for i, (seed, zp, folder) in enumerate(all_seeds):
        genotype = parse_name(seed)
        try:
            imgs, df = load_frames_and_csv(zp, folder)
        except Exception as e:
            print(f"  [skip] {seed}: load failed ({e})")
            continue
        if not imgs or df is None or df.empty:
            print(f"  [skip] {seed}: no frames or empty csv")
            continue

        lengths = df["Length"].astype(float).values
        n_fr = min(len(imgs), len(lengths))
        # Pick BOTH lateral views: max-Length frame + the one 180° opposite
        max_k = int(np.argmax(lengths[:n_fr]))
        opp_k = (max_k + n_fr // 2) % n_fr

        profs, tilts, lengths_mm = [], [], []
        for k in (max_k, opp_k):
            if k not in imgs:
                continue
            mask = segment(imgs[k])
            mask = trim_narrow_rows(mask, min_width_px=20)
            mask = trim_thin_tails(mask, min_frac=0.2)
            if not mask.any():
                continue
            tilt = principal_axis_angle(mask)
            mask_rot = rotate_mask(mask, tilt)
            gray_rot = rotate_gray(cv2.cvtColor(imgs[k], cv2.COLOR_BGR2GRAY),
                                   mask, tilt)
            xs_r = np.where(mask_rot.any(axis=0))[0]
            if len(xs_r) < MIN_LEN_PX:
                continue
            length_px_rot = xs_r.max() - xs_r.min() + 1
            mm_per_px = lengths[k] / length_px_rot
            prof, length_mm = thickness_profile(mask_rot, mm_per_px)
            if prof is None:
                continue
            prof = orient_embryo_left(prof, mask_rot, gray_rot, end_frac=0.35)
            profs.append(prof)
            tilts.append(tilt)
            lengths_mm.append(length_mm)

        if not profs:
            print(f"  [skip] {seed}: no usable lateral view")
            continue

        # Average the two views (or use the single one if only one was usable)
        prof      = np.mean(profs, axis=0)
        tilt      = float(np.mean(np.abs(tilts)))    # mean |tilt| across views
        length_mm = float(np.mean(lengths_mm))

        row = {"seed": seed, "genotype": genotype,
               "length_mm": length_mm, "tilt_deg": tilt,
               "n_views": len(profs)}                  # NEW: track how many
        for k in range(N_PROF):
            row[f"t_{k:03d}"] = prof[k]
        rows.append(row)

        if (i + 1) % 20 == 0 or i + 1 == len(all_seeds):
            print(f"  {i + 1}/{len(all_seeds)} processed, kept {len(rows)}")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} seeds × ({N_PROF} positions + 4 metadata cols) "
          f"to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory containing the per-seed .zip bundles "
                         "(default: directory of this script)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: <data_dir>/"
                         "thickness_profiles_per_seed.csv)")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.data_dir,
                                        "thickness_profiles_per_seed.csv")
    main(args.data_dir, out_path)
