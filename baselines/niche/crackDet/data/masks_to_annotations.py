"""Auto-generate oriented sub-crack box annotations from this project's existing pixel
masks, across ALL 7 source datasets and ALL defect classes -- not filtered by whether a
class is literally named "crack". CrackDet's architecture is shape-based (does a defect
benefit from being represented as a chain of small oriented rectangles?), not label-based;
a "weld line" or "defect3" streak is exactly the kind of elongated defect this model
targets, whatever it's called.

IMPORTANT: this is heuristic auto-labeling, NOT ground truth, and NOT something the paper
describes at all -- CrackDet's own datasets (ONPP/ORC/OCCSD) were hand-annotated with
oriented boxes from the start; there is no published procedure for deriving them from a
pixel mask. This script is a practical bridge to get *some* real, shape-derived training
signal out of data this project already has (pixel masks), instead of either hand-labeling
from scratch or leaving CrackDet untrainable. Spot-check the output against a few source
masks before trusting it at scale -- this has never been run (see repo README: no local
Python runtime in this session).

Algorithm, per connected component (defect blob) in a mask:
  1. Compact/round blobs (bounding-box aspect ratio < 2.5, or too small to meaningfully
     skeletonize) get ONE box for the whole blob via cv2.minAreaRect. This is intentional,
     not a cop-out: a defect with no clear elongation has no clear "sub-crack" angle either,
     which is precisely the orientation-ambiguity case the paper's own MAR loss is designed
     to handle (Fig. 1c) via a high predicted variance -- not something this converter
     should fake precision on.
  2. Elongated blobs get skeletonized (skimage.morphology.skeletonize -> a 1px-wide
     centerline), the skeleton is walked into an ordered path from an endpoint via greedy
     nearest-neighbor chaining (stops rather than jumping across a branch point -- so on a
     branching/Y-shaped skeleton this only captures the first-explored branch; a disclosed
     limitation, not a silent one), and the path is chunked into fixed-length segments.
     Each segment becomes one oriented box: center = segment centroid, angle = the
     segment's own best-fit direction (least-squares line through its points, not
     `cv2.minAreaRect` on the raw 1px-wide skeleton points, which degenerates to ~0 width),
     length = segment arc length, width = local stroke thickness estimated via
     `cv2.distanceTransform` (distance-to-background, doubled, at the segment's midpoint --
     a standard technique for estimating stroke width from a skeleton).

Usage:
    python masks_to_annotations.py --dataset-root ../../../processed_output \
        --output raw_annotations.json [--datasets MTD Severstal] [--classes crack defect1 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

MIN_BLOB_AREA = 15          # px^2, drop smaller specks as mask noise
ELONGATION_THRESHOLD = 2.5  # bbox aspect ratio above which a blob gets skeletonized
SEGMENT_LENGTH_PX = 40.0    # target arc length per oriented sub-box
MAX_WALK_STEP_PX = 4.0      # stop the skeleton walk rather than jump across a gap/branch


def find_matching_image(mask_path: Path, image_dir: Path) -> Path | None:
    """Mirrors the candidate-stem matching already used elsewhere in this project's RunPod
    scripts (mask filename -> defect image filename don't always share an exact stem)."""
    stem = mask_path.stem  # e.g. "mtd_crack_0001_mask"
    base = stem.removesuffix("_mask")
    candidates = [stem, base, base + "_defect", base + "_img"]
    for cand in candidates:
        for ext in IMAGE_EXTS:
            candidate_path = image_dir / f"{cand}{ext}"
            if candidate_path.exists():
                return candidate_path
    return None


def _fit_box(points: np.ndarray, width: float):
    """Fit a rotated box to a set of (x, y) points: center = centroid, angle = best-fit
    direction (largest-eigenvalue eigenvector of the point covariance), length = point
    spread along that direction, width = the given (externally estimated) stroke width."""
    if len(points) < 2:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    if np.isscalar(cov) or cov.shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]  # unit vector along the segment's main axis

    projections = centered @ direction
    length = float(projections.max() - projections.min())
    if length < 1.0:
        return None

    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    return {
        "cx": float(centroid[0]), "cy": float(centroid[1]),
        "h": max(length, 1.0), "w": max(width, 1.0),
        "theta_deg": angle_deg,
    }


def _order_skeleton_path(skel_points: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor walk from an endpoint. Stops (rather than jumping) once the
    nearest remaining point is farther than MAX_WALK_STEP_PX -- on a branching skeleton this
    only follows the first-explored branch; the rest is simply not converted to boxes."""
    remaining = skel_points.copy()
    # Prefer an actual endpoint (fewest neighbors within 1.5px) as the start; fall back to
    # the point farthest from the centroid if the skeleton is a loop with no clear endpoint.
    from scipy.spatial import cKDTree
    tree = cKDTree(remaining)
    neighbor_counts = tree.query_ball_point(remaining, r=1.5, return_length=True) - 1
    endpoint_idx = np.where(neighbor_counts == 1)[0]
    if len(endpoint_idx) > 0:
        start_idx = endpoint_idx[0]
    else:
        centroid = remaining.mean(axis=0)
        start_idx = int(np.argmax(np.linalg.norm(remaining - centroid, axis=1)))

    ordered = [remaining[start_idx]]
    used = np.zeros(len(remaining), dtype=bool)
    used[start_idx] = True
    current = remaining[start_idx]

    while True:
        unused_idx = np.where(~used)[0]
        if len(unused_idx) == 0:
            break
        dists = np.linalg.norm(remaining[unused_idx] - current, axis=1)
        nearest = unused_idx[np.argmin(dists)]
        if dists.min() > MAX_WALK_STEP_PX:
            break
        used[nearest] = True
        current = remaining[nearest]
        ordered.append(current)

    return np.array(ordered)


def mask_to_oriented_boxes(mask: np.ndarray) -> list[dict]:
    """`mask` is a 2D array, nonzero = defect pixel. Returns a list of
    {"cx", "cy", "h", "w", "theta_deg", "label"} dicts in image-pixel coordinates."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    boxes = []
    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        area = int(component.sum())
        if area < MIN_BLOB_AREA:
            continue

        ys, xs = np.nonzero(component)
        bbox_h, bbox_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)

        if aspect < ELONGATION_THRESHOLD:
            points = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(points)
            (cx, cy), _, _ = rect
            box_pts = cv2.boxPoints(rect)
            # Derive h (paper's long side, Fig. 2's "long-side definition"), w (short side),
            # and theta_deg (the long side's own direction) directly from the box corners --
            # NOT from rect's (width, height, angle) tuple, whose angle-range convention and
            # corner-ordering relative to (width, height) differs across OpenCV versions
            # (pre-/post-4.5). Two adjacent box corners always share the edge length that's
            # actually consistent with that edge's own direction, regardless of OpenCV
            # version, so this avoids the mismatch risk entirely.
            edge_ab = box_pts[1] - box_pts[0]
            edge_bc = box_pts[2] - box_pts[1]
            len_ab, len_bc = float(np.linalg.norm(edge_ab)), float(np.linalg.norm(edge_bc))
            long_edge, h_len, w_len = (edge_ab, len_ab, len_bc) if len_ab >= len_bc else (edge_bc, len_bc, len_ab)
            angle_deg = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])) % 180.0)
            boxes.append({"cx": float(cx), "cy": float(cy), "h": max(h_len, 1.0),
                           "w": max(w_len, 1.0), "theta_deg": angle_deg, "label": 0})
            continue

        skeleton = skeletonize(component.astype(bool))
        sk_ys, sk_xs = np.nonzero(skeleton)
        if len(sk_xs) < 2:
            continue
        skel_points = np.column_stack([sk_xs, sk_ys]).astype(np.float64)

        ordered = _order_skeleton_path(skel_points)
        if len(ordered) < 2:
            continue

        arc_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1))])
        total_length = arc_lengths[-1]
        n_segments = max(1, round(total_length / SEGMENT_LENGTH_PX))
        boundaries = np.linspace(0, total_length, n_segments + 1)

        for seg_idx in range(n_segments):
            seg_mask = (arc_lengths >= boundaries[seg_idx]) & (arc_lengths <= boundaries[seg_idx + 1])
            seg_points = ordered[seg_mask]
            if len(seg_points) < 2:
                continue
            mid_xy = seg_points[len(seg_points) // 2].astype(int)
            local_width = 2.0 * float(dist_transform[min(mid_xy[1], dist_transform.shape[0] - 1),
                                                       min(mid_xy[0], dist_transform.shape[1] - 1)])
            box = _fit_box(seg_points, width=max(local_width, 2.0))
            if box is not None:
                box["label"] = 0
                boxes.append(box)

    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="path to processed_output/")
    parser.add_argument("--output", required=True, help="output raw_annotations.json path")
    parser.add_argument("--datasets", nargs="*", default=DATASET_NAMES,
                         help=f"subset of {DATASET_NAMES} to include (default: all)")
    parser.add_argument("--classes", nargs="*", default=None,
                         help="only include mask files whose filename contains one of these "
                              "class-name substrings, e.g. --classes crack defect1 defect2 "
                              "2_hanfeng (default: every class in every dataset)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    records = []
    total_masks = total_boxes = skipped_no_image = skipped_no_boxes = 0

    for dataset_name in args.datasets:
        for size_bucket in SIZE_BUCKETS:
            mask_dir = dataset_root / dataset_name / size_bucket / "masks"
            image_dir = dataset_root / dataset_name / size_bucket / "images"
            if not mask_dir.exists():
                continue

            for mask_path in sorted(mask_dir.glob("*.png")):
                if mask_path.name.startswith("._"):  # macOS AppleDouble junk
                    continue
                if args.classes and not any(c in mask_path.stem for c in args.classes):
                    continue

                image_path = find_matching_image(mask_path, image_dir)
                if image_path is None:
                    skipped_no_image += 1
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    skipped_no_image += 1
                    continue

                total_masks += 1
                boxes = mask_to_oriented_boxes(mask)
                if not boxes:
                    skipped_no_boxes += 1
                    continue

                records.append({
                    "image": str(image_path), "dataset": dataset_name, "size_bucket": size_bucket,
                    "boxes": boxes,
                })
                total_boxes += len(boxes)

    with open(args.output, "w") as f:
        json.dump(records, f)

    print(f"Scanned {total_masks} mask(s) across {len(args.datasets)} dataset(s).")
    print(f"  {len(records)} image(s) with >=1 box, {total_boxes} box(es) total.")
    print(f"  Skipped: {skipped_no_image} (no matching image found), "
          f"{skipped_no_boxes} (mask produced 0 boxes -- likely below MIN_BLOB_AREA={MIN_BLOB_AREA}).")
    print(f"Written to {args.output}")
    print("\nThis is heuristic auto-labeling, not verified ground truth -- spot-check a few "
          "entries against their source masks before trusting this at scale.")


if __name__ == "__main__":
    main()
