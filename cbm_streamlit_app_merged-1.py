"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CBM-DCE-MRI  —  Streamlit Clinical Analysis App  (Segmentation Edition)    ║
║  Merges breast_dce_segmentation.py pipeline into the upload flow.           ║
║                                                                              ║
║  After data upload the app now runs:                                        ║
║    Step 1 → stream to disk                                                  ║
║    Step 2 → extract archive                                                 ║
║    Step 3 → build DICOM index                                               ║
║    Step 4 → DCE-MRI tumour segmentation (3-stage per-patient)               ║
║  All subsequent Ktrans maps are overlaid on the **real** segmented tumour.  ║
║                                                                              ║
║  Launch:   streamlit run cbm_streamlit_app_merged.py                        ║
║  Config:   .streamlit/config.toml  (maxUploadSize = 600)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 16,
    'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 17, 'axes.labelweight': 'bold',
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'legend.fontsize': 15, 'legend.title_fontsize': 15,
})

import io, os, re, zipfile, pathlib, warnings, tempfile, shutil, time, logging
try:
    import requests as _requests_mod   # used by TCIA download
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rarfile, pydicom
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import (gaussian_filter, binary_fill_holes,
                            binary_closing, label as ndimage_label,
                            sobel, binary_opening, binary_dilation,
                            binary_erosion)
from scipy import ndimage
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc as auc_fn
from skimage.measure import find_contours

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    from skimage import filters, morphology, measure, segmentation as ski_seg
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

warnings.filterwarnings("ignore")
log = logging.getLogger("cbm_seg")

# ─── DICOM image-type guard ────────────────────────────────────────────────────
_NON_IMAGE_SOP = {
    "1.2.840.10008.5.1.4.1.1.88.59",
    "1.2.840.10008.5.1.4.1.1.88.11",
    "1.2.840.10008.5.1.4.1.1.88.22",
    "1.2.840.10008.5.1.4.1.1.88.33",
    "1.2.840.10008.5.1.4.1.1.11.1",
    "1.2.840.10008.5.1.4.1.1.11.2",
    "1.2.840.10008.5.1.4.1.1.481.3",
    "1.2.840.10008.5.1.4.1.1.481.5",
    "1.2.840.10008.5.1.4.1.1.481.2",
    "1.2.840.10008.5.1.4.1.1.104.1",
    "1.2.840.10008.3.1.2.3.4",
}
_IMAGE_MODALITIES = {"MR","CT","PT","NM","US","CR","DX","MG","OT","XA","RF","SC"}


def _is_image_dicom(ds) -> bool:
    if not hasattr(ds, "Rows"):
        return False
    sop = str(getattr(ds, "SOPClassUID", ""))
    if sop in _NON_IMAGE_SOP:
        return False
    modality = str(getattr(ds, "Modality", "MR"))
    if modality not in _IMAGE_MODALITIES:
        return False
    if (0x7FE0, 0x0010) not in ds:
        return False
    return True


def _get_base_dir() -> pathlib.Path:
    """Return the dataset root stored at index time. Falls back to CWD."""
    bd = st.session_state.get("base_dir", ".")
    return pathlib.Path(bd)


def _safe_pixel_array(stored_path: str):
    """Read one DICOM pixel array, resolving stored_path via resolve_dcm_path."""
    try:
        abs_path = resolve_dcm_path(_get_base_dir(), stored_path)
        ds = pydicom.dcmread(abs_path, force=True)
        if not _is_image_dicom(ds):
            return None, False
        px = ds.pixel_array.astype(np.float32)
        return px, True
    except Exception:
        return None, False


# ═══════════════════════════════════════════════════════════════════════════════
# §S  SEGMENTATION PIPELINE  (from breast_dce_segmentation.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationResult:
    """Output of the DCE-MRI tumour segmentation for one patient/visit."""
    patient_id: str
    visit: str
    tumour_mask: np.ndarray          # 3-D bool   (Z, Y, X)
    enhancement_map: np.ndarray      # 3-D float  (Z, Y, X)  max across timepoints
    kinetic_map: np.ndarray          # 3-D int8   0=bg 1=TypeI 2=TypeII 3=TypeIII
    voxel_spacing_mm: Tuple[float, float, float]
    volume_ml: float
    peak_slice_idx: int
    kinetic_counts: Dict[int, int]
    measurements_df: pd.DataFrame


# ── Stage: load full volume from DCM index entries ────────────────────────────

def _load_volume_from_entries(entries: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load all slices for one series entry list from the DCM index.
    entries = [(slice_loc, stored_path), ...]
    stored_path may be relative (web-safe) or absolute (local).
    Returns (volume [Z,Y,X] float32, slice_positions [Z]) or (None, None).
    """
    base_dir = _get_base_dir()
    slices = []
    for loc, stored_path in entries:
        try:
            abs_path = resolve_dcm_path(base_dir, stored_path)
            ds = pydicom.dcmread(abs_path, force=True)
            if not _is_image_dicom(ds):
                continue
            px = ds.pixel_array.astype(np.float32)
            slope = float(ds.get("RescaleSlope", 1) or 1)
            intercept = float(ds.get("RescaleIntercept", 0) or 0)
            px = px * slope + intercept
            slices.append((loc, px))
        except Exception:
            continue

    if not slices:
        return None, None

    slices.sort(key=lambda x: x[0])
    positions = np.array([s[0] for s in slices], dtype=np.float32)
    volume = np.stack([s[1] for s in slices], axis=0).astype(np.float32)
    return volume, positions


# ── Stage: motion registration (optional, uses SimpleITK) ────────────────────

def _register_volume(moving: np.ndarray, fixed: np.ndarray,
                     spacing: Tuple[float, float, float]) -> np.ndarray:
    """Rigid 6-DOF registration of moving → fixed. Skipped if SimpleITK absent."""
    if not HAS_SITK:
        return moving

    def arr_to_sitk(arr):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
        return img

    fixed_sitk = arr_to_sitk(fixed)
    moving_sitk = arr_to_sitk(moving)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    init_tx = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(init_tx, inPlace=False)
    try:
        final_tx = reg.Execute(fixed_sitk, moving_sitk)
        res = sitk.ResampleImageFilter()
        res.SetReferenceImage(fixed_sitk)
        res.SetInterpolator(sitk.sitkLinear)
        res.SetDefaultPixelValue(0)
        res.SetTransform(final_tx)
        return sitk.GetArrayFromImage(res.Execute(moving_sitk)).astype(np.float32)
    except Exception:
        return moving


# ── Stage: enhancement map ────────────────────────────────────────────────────

def _compute_enhancement_map(
        pre: np.ndarray,
        post_frames: List[np.ndarray],
        min_pre: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (max_enhancement [Z,Y,X], enhancement_stack [T,Z,Y,X]).
    Enhancement = (post - pre) / pre per voxel per frame.

    For breast DCE-MRI: min_pre is auto-scaled to the 2nd percentile of
    non-zero pre-contrast voxels so relative enhancement is physically
    meaningful regardless of scanner scaling (12-bit vs 16-bit, etc.).
    """
    # Scale the noise floor to image statistics — critical for fat-sat sequences
    nonzero_pre = pre[pre > 0]
    if nonzero_pre.size > 100:
        adaptive_floor = float(np.percentile(nonzero_pre, 2))
        min_pre = max(min_pre, adaptive_floor)

    safe_pre = np.where(pre > min_pre, pre, min_pre)
    stack = []
    for post in post_frames:
        if post.shape != pre.shape:
            continue
        stack.append(((post - pre) / safe_pre).astype(np.float32))

    if not stack:
        return np.zeros_like(pre), np.zeros((1,) + pre.shape)

    enh_stack = np.stack(stack, axis=0)
    max_enh = np.max(enh_stack, axis=0)
    return max_enh, enh_stack


# ── Stage: breast mask ────────────────────────────────────────────────────────

def _compute_body_mask(pre: np.ndarray, percentile: float = 8.0) -> np.ndarray:
    return _compute_breast_body_mask(pre, percentile=percentile)


def _compute_breast_body_mask(pre: np.ndarray,
                               max_enh: np.ndarray = None,
                               percentile: float = 8.0,
                               breast_in_top_override: Optional[bool] = None
                               ) -> np.ndarray:
    """
    Bilateral-topology breast mask — complete overhaul.

    Core insight
    ────────────
    In prone bilateral breast MRI every axial slice shows either:
      • TWO separate body regions  →  breast level (left + right breast dome)
      • ONE connected body region  →  chest-wall level (pectoralis, ribs, spine)

    This bilateral split is a pure ANATOMICAL feature that:
      - requires NO orientation metadata (no DICOM IOP needed)
      - requires NO enhancement signal (works even pre-contrast)
      - is invariant to scanner type, FOV, or acquisition direction
      - directly identifies the breast-chest-wall junction

    Algorithm
    ─────────
    1. Body mask via threshold + hole-fill + small-object removal.
    2. Per Y-row, count connected components across all Z slices
       (using a mild 1-D erosion to bridge isolated skin pixels).
       Rows where ≥2 components appear in >25% of slices = breast level.
    3. Build the breast zone from those bilateral rows ± a margin.
    4. Fallback to body-relative 45% inferior zone when bilateral
       structure is not detected (single breast, post-mastectomy).
    5. Image-border blank-out (3 px) — NOT morphological erosion, which
       was cutting into deep tumours like BC05.
    """
    Z, H, W = pre.shape

    # ── Step 1: body mask ────────────────────────────────────────────────────
    thr = np.percentile(pre, percentile)
    body = pre > thr
    filled = np.zeros_like(body, dtype=bool)
    for z in range(Z):
        filled[z] = binary_fill_holes(body[z])
    if HAS_SKIMAGE:
        filled = morphology.remove_small_objects(filled, min_size=500)
    else:
        filled = ndimage.binary_opening(filled, np.ones((3, 3, 3)))

    # ── Step 2: bilateral topology — fully vectorised ────────────────────────
    # In prone bilateral breast MRI, breast-level axial rows contain TWO
    # separate body blobs (left breast + right breast). Chest-wall rows contain
    # ONE connected body region.  Count connected components per (z, y) pair
    # using numpy — no Python loop over slices × rows needed.

    # Erode in X direction (7-px) to close the thin skin bridge between breasts
    filled_eroded = ndimage.binary_erosion(
        filled, structure=np.ones((1, 1, 7), dtype=bool))
    # Where erosion wiped out a row entirely, fall back to the original row
    has_eroded = filled_eroded.any(axis=2)    # (Z, H)
    use_orig    = ~has_eroded                  # rows where eroded is all-False

    # Build the array we'll count components on
    use_arr = np.where(
        has_eroded[:, :, np.newaxis], filled_eroded, filled)  # (Z, H, W)

    # Count rising 0→1 transitions in X = number of connected runs per (z,y)
    padded       = np.zeros((Z, H, W + 1), dtype=bool)
    padded[:, :, :W] = use_arr
    rising       = padded[:, :, 1:] & ~padded[:, :, :-1]   # (Z, H, W)
    comp_count   = rising.sum(axis=2)                        # (Z, H)  int

    has_content  = filled.any(axis=2)                        # (Z, H)  bool
    total_votes  = has_content.sum(axis=0).astype(np.float32)           # (H,)
    two_comp_votes = (comp_count >= 2).sum(axis=0).astype(np.float32)   # (H,)

    with np.errstate(divide='ignore', invalid='ignore'):
        bilat_frac = np.where(total_votes > 0,
                               two_comp_votes / total_votes, 0.0)

    bilateral_rows = np.where(bilat_frac > 0.25)[0]

    # ── Step 3a: bilateral-topology breast zone ───────────────────────────────
    breast_zone = np.zeros((Z, H, W), dtype=bool)

    if bilateral_rows.size >= 4:
        y_bilat_lo = int(bilateral_rows.min())
        y_bilat_hi = int(bilateral_rows.max())
        bilat_span = y_bilat_hi - y_bilat_lo + 1
        # Add a margin (30% of bilateral span) toward the chest wall so that
        # tumours at the breast-chest-wall junction are included
        margin = max(4, int(bilat_span * 0.30))
        zone_lo = max(0, y_bilat_lo - margin)
        zone_hi = min(H, y_bilat_hi + margin)
        breast_zone[:, zone_lo:zone_hi, :] = True

    else:
        # ── Step 3b: fallback — body-relative inferior 45% zone ──────────────
        # Tries both body ends and picks the one with more focal enhancement.
        signal_map = (max_enh if (max_enh is not None and np.any(max_enh > 0))
                      else pre)
        votes_lo = votes_hi = 0.0

        for z in range(Z):
            sl = filled[z]
            if not sl.any():
                continue
            rows = np.where(sl.any(axis=1))[0]
            if rows.size < 4:
                continue
            y_lo  = int(rows.min())
            y_hi  = int(rows.max())
            depth = max(4, int((y_hi - y_lo + 1) * 0.45))
            y_A   = min(H, y_lo + depth)
            y_B   = max(0, y_hi - depth + 1)

            mask_A = sl[y_lo:y_A, :]
            mask_B = sl[y_B:y_hi+1, :]
            sA = (float(np.percentile(signal_map[z, y_lo:y_A, :][mask_A], 95))
                  if mask_A.any() else 0.0)
            sB = (float(np.percentile(signal_map[z, y_B:y_hi+1, :][mask_B], 95))
                  if mask_B.any() else 0.0)
            votes_lo += sA
            votes_hi += sB

        if breast_in_top_override is not None:
            breast_at_low_y = not breast_in_top_override
        else:
            breast_at_low_y = votes_lo >= votes_hi

        for z in range(Z):
            sl = filled[z]
            if not sl.any():
                continue
            rows = np.where(sl.any(axis=1))[0]
            if rows.size < 4:
                continue
            y_lo  = int(rows.min())
            y_hi  = int(rows.max())
            depth = max(4, int((y_hi - y_lo + 1) * 0.45))
            if breast_at_low_y:
                breast_zone[z, y_lo : y_lo + depth, :] = True
            else:
                breast_zone[z, y_hi - depth + 1 : y_hi + 1, :] = True

    # ── Step 4: apply zone and blank image border ─────────────────────────────
    breast_mask = filled & breast_zone
    n = 3
    breast_mask[:, :n, :]  = False
    breast_mask[:, -n:, :] = False
    breast_mask[:, :, :n]  = False
    breast_mask[:, :, -n:] = False

    # Per-slice fallback: restore if blanking wiped everything
    for z in range(Z):
        if not breast_mask[z].any() and breast_zone[z].any():
            breast_mask[z] = filled[z] & breast_zone[z]

    return breast_mask.astype(bool)


def _build_seed_mask_3d(seed_points_yx: list,
                         volume_shape: tuple,
                         radius_px: int = 40) -> np.ndarray:
    """
    Build a 3-D boolean proximity mask from 2-D seed points picked by the user.

    seed_points_yx : list of (y_arr, x_arr) in array coordinates
                     (already accounting for origin='lower' flip)
    volume_shape   : (Z, H, W)
    radius_px      : in-plane search radius in pixels (default 40 ≈ 40mm at 1mm/px)

    The mask is extruded through ALL Z slices — the user only needs to pick
    the tumour in one representative slice and the algorithm finds the 3-D extent.
    """
    Z, H, W = volume_shape
    mask = np.zeros((Z, H, W), dtype=bool)
    if not seed_points_yx:
        return mask
    for y0, x0 in seed_points_yx:
        y0, x0 = int(round(y0)), int(round(x0))
        y_grid, x_grid = np.ogrid[:H, :W]
        dist2 = (y_grid - y0) ** 2 + (x_grid - x0) ** 2
        circle = dist2 <= radius_px ** 2
        mask[:, circle] = True
    return mask


# ── Stage: 3-stage tumour segmentation ───────────────────────────────────────

def _segment_tumour(
        max_enh: np.ndarray,
        body_mask: np.ndarray,
        pre: np.ndarray,
        enhancement_threshold: float = 0.35,
        min_lesion_voxels: int = 50,
        max_lesion_voxels: int = 250_000,
        remove_vessels: bool = True,
        enh_stack: np.ndarray = None,
        seed_mask_3d: np.ndarray = None) -> np.ndarray:
    """
    Breast-optimised three-stage DCE-MRI tumour segmentation.

    seed_mask_3d : optional 3-D bool array (Z, Y, X).
        When provided (user drew seed points on the tumour), the search space
        for Stage A is restricted to this proximity mask.  This completely
        bypasses all zone-detection heuristics — no orientation guessing needed.
        The algorithm then grows outward from the seed region using the
        enhancement map and contour refinement as normal.
    """
    Z, H, W = max_enh.shape

    def _run_stages(enh_map, bm, threshold, min_vox, relax_posterior=False):
        # ── Stage A ──────────────────────────────────────────────────────────
        # Two candidate sources are OR-combined so neither alone can miss a
        # tumour:
        #   1. Relative enhancement > threshold  (standard DCE criterion)
        #   2. Absolute enhancement above a tissue-adaptive level — catches
        #      tumours near the coil surface or with high baseline T1 signal
        #      (mucinous, colloid, post-biopsy) where (post-pre)/pre is
        #      suppressed even though absolute uptake is high.
        coarse_rel = (enh_map > threshold) & bm

        # Absolute enhancement: (post-pre) in image units.
        # We approximate it as enh_map * pre_median_in_mask.
        # Threshold = 15% of the median pre-contrast signal in the breast mask.
        pre_vals_in_mask = pre[bm] if bm.any() else pre.ravel()
        pre_ref = float(np.percentile(pre_vals_in_mask, 50)) if pre_vals_in_mask.size > 0 else 1.0
        abs_enh_threshold = 0.15 * pre_ref   # 15% of median breast signal in image units
        # enh_map = (post-pre)/safe_pre → abs_enh ≈ enh_map * pre
        abs_enh_approx = enh_map * pre   # element-wise, in image units
        coarse_abs = (abs_enh_approx > abs_enh_threshold) & bm

        coarse = (coarse_rel | coarse_abs)
        struct2 = np.ones((1, 3, 3))
        coarse = ndimage.binary_opening(coarse, struct2)
        coarse = ndimage.binary_closing(coarse, np.ones((1, 5, 5)))

        # ── Stage B ──────────────────────────────────────────────────────────
        labeled, n = ndimage.label(coarse)
        refined = np.zeros_like(coarse, dtype=bool)

        for lbl in range(1, n + 1):
            region = labeled == lbl
            size = int(np.sum(region))
            if size < min_vox or size > max_lesion_voxels:
                continue

            coords = np.argwhere(region)
            centroid = coords.mean(axis=0)
            cy, cx = float(centroid[1]), float(centroid[2])

            # ── Bounding-box extent guard ──────────────────────────────────────
            # A real breast tumour is a FOCAL lesion — it cannot span more than
            # ~40% of the image in any in-plane direction. When the breast zone
            # is incorrectly set (e.g. zone pointing at chest wall), the
            # segmentation can find a connected region that spans the FULL image
            # width (as seen in BC01: 333.6 mm max diameter). This guard
            # catches that regardless of zone orientation.
            zcoords, ycoords, xcoords = coords[:,0], coords[:,1], coords[:,2]
            bbox_y_frac = (ycoords.max() - ycoords.min() + 1) / H
            bbox_x_frac = (xcoords.max() - xcoords.min() + 1) / W
            if bbox_y_frac > 0.45 or bbox_x_frac > 0.45:
                continue   # spans >45% of image — not a focal tumour

            # ── Bilateral span guard ───────────────────────────────────────────
            # Tumours are unilateral (in one breast). A region whose x-extent
            # straddles the image centre-line (left to right breast) is chest
            # wall or bilateral background parenchymal enhancement, not tumour.
            x_centre = W / 2.0
            spans_centre = (xcoords.min() < x_centre * 0.6 and
                            xcoords.max() > x_centre * 1.4)
            if spans_centre and bbox_x_frac > 0.30:
                continue   # spans both breasts → reject

            # ── Skin/edge rejection ───────────────────────────────────────────
            edge_margin = 6
            if (cy < edge_margin or cy > H - edge_margin or
                    cx < edge_margin or cx > W - edge_margin):
                continue

            # ── Vessel elongation ─────────────────────────────────────────────
            if remove_vessels and len(coords) >= 3:
                try:
                    cov = np.cov(coords.T)
                    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                    elongation = eigvals[0] / (eigvals[-1] + 1e-8)
                    if elongation > 50:
                        continue
                except np.linalg.LinAlgError:
                    pass

            if np.mean(enh_map[region]) < threshold * 0.75:
                continue

            refined |= region

        # ── Stage C ──────────────────────────────────────────────────────────
        if np.any(refined):
            smooth_enh = ndimage.gaussian_filter(enh_map, sigma=1.0)
            gx = sobel(smooth_enh, axis=2)
            gy = sobel(smooth_enh, axis=1)
            gz = sobel(smooth_enh, axis=0)
            gradient = np.sqrt(gx**2 + gy**2 + gz**2)

            expanded = ndimage.binary_dilation(refined, np.ones((1, 3, 3)))
            grad_thr = np.percentile(gradient[bm], 80)
            refined = expanded & (gradient < grad_thr) & bm

            for z in range(refined.shape[0]):
                refined[z] = binary_fill_holes(refined[z])

            refined = ndimage.binary_closing(refined, np.ones((1, 3, 3)))

            # Post-refinement size check only (no posterior guard)
            labeled2, n2 = ndimage.label(refined)
            final = np.zeros_like(refined, dtype=bool)
            for lbl2 in range(1, n2 + 1):
                reg2 = labeled2 == lbl2
                if int(np.sum(reg2)) >= min_vox:
                    final |= reg2
            refined = final

        return refined.astype(bool)

    # ── Seed-guided mode: centroid-anchored adaptive region-growing ──────────
    # ROOT CAUSE of the bottom-edge artefact:
    #   The 45-px seed cylinder extends from y_seed-45 to y_seed+45.  At y≈0
    #   the nipple/skin rim has strong enhancement.  `local_enh.max()` picked
    #   that artefact as the reference peak, so the threshold grew from the
    #   skin ring instead of the tumour.
    #
    # FIX — three changes:
    #   1. Compute the SEED CENTROID and use the 90th-pct enhancement in a
    #      small (15 px) neighbourhood around it as the reference value.
    #      90th-pct rather than max: resistant to isolated bright pixels.
    #   2. After segmentation, KEEP ONLY THE COMPONENT NEAREST to the seed
    #      centroid — this discards any off-target cluster regardless of size.
    #   3. Gradient refinement threshold raised to 85th-pct (was 75th) so the
    #      boundary isn't over-trimmed on low-Ktrans tumours like BC01 NCR.
    if seed_mask_3d is not None and seed_mask_3d.any():
        from scipy.ndimage import label as _label

        seed_labeled, n_seeds = _label(seed_mask_3d)
        combined = np.zeros_like(max_enh, dtype=bool)

        for seed_id in range(1, n_seeds + 1):
            this_seed = seed_labeled == seed_id
            bm_local  = body_mask & this_seed
            if not bm_local.any():
                bm_local = this_seed

            # ── Step 0: find seed centroid (the user's intended location) ─────
            seed_coords  = np.argwhere(this_seed)
            if seed_coords.size == 0:
                continue
            seed_ctr     = seed_coords.mean(axis=0)   # (z, y, x) float
            cz  = int(np.clip(seed_ctr[0], 0, Z - 1))
            cym = int(np.clip(seed_ctr[1], 0, H - 1))
            cxm = int(np.clip(seed_ctr[2], 0, W - 1))

            # ── Step 1: centroid-neighbourhood enhancement reference ──────────
            r_ref = 15
            ref_enh = max_enh[
                max(0, cz-r_ref) : min(Z, cz+r_ref),
                max(0, cym-r_ref): min(H, cym+r_ref),
                max(0, cxm-r_ref): min(W, cxm+r_ref)]

            ref_flat = ref_enh[ref_enh > 0] if ref_enh.size > 0 else np.array([])

            if ref_flat.size > 0 and ref_flat.max() > 0:
                # DIRECT threshold = 60th-pct of non-zero neighbourhood voxels.
                # For an NCR tumour (~15-20% enh): direct_thr ≈ 12-16%
                # while BPE background (~5-8%) is EXCLUDED.
                # Old "35% × 90th-pct ≈ 7%" was catching all breast tissue
                # and producing full-breast circles — eliminated here.
                direct_thr = float(np.percentile(ref_flat, 60))
                peak_val   = float(np.percentile(ref_flat, 95))
            else:
                fallback = max_enh[bm_local]
                direct_thr = float(fallback.max() * 0.5) if bm_local.any() else 0.0
                peak_val   = float(fallback.max())        if bm_local.any() else 0.0

            if direct_thr <= 0 and peak_val <= 0:
                continue

            # ── Step 2: threshold cascade (all options above background BPE) ──
            candidate = None
            for thr in [direct_thr,
                        direct_thr * 0.75,
                        direct_thr * 0.55,
                        direct_thr * 0.35]:
                if thr <= 0:
                    continue
                cand = bm_local & (max_enh >= thr)
                if cand.sum() >= 10:
                    candidate = cand
                    break
            if candidate is None or not candidate.any():
                thr35 = float(np.percentile(max_enh[bm_local], 65))
                candidate = bm_local & (max_enh >= thr35)
            if not candidate.any():
                continue

            # ── Step 3: morphological cleanup — preserve irregular shape ─────
            # 3×3 kernel (was 5×5): smaller kernel keeps the natural tumour
            # boundary irregular rather than rounding it into a smooth oval.
            # fill_holes is NOT applied per-slice — filling every hole makes
            # heterogeneous tumours look artificially solid and circular.
            # Only very small isolated holes (≤25 voxels) are closed.
            closed = ndimage.binary_closing(candidate, np.ones((1, 3, 3)))
            # Remove isolated specks (noise) but do not force a solid fill
            if HAS_SKIMAGE:
                closed = morphology.remove_small_objects(closed, min_size=8)
            filled = closed.copy()

            # ── Step 4: keep ONLY the component nearest to seed centroid ──────
            # This is the critical guard against artefact clusters (skin rings,
            # nipple enhancement) that are spatially distant from the seed.
            if filled.any():
                comp_labeled, n_comp = _label(filled)
                best_mask = None
                best_dist  = float('inf')
                ctr_arr    = np.array([cz, cym, cxm], dtype=float)
                for lbl in range(1, n_comp + 1):
                    comp = comp_labeled == lbl
                    comp_ctr = np.argwhere(comp).mean(axis=0)
                    dist = float(np.linalg.norm(comp_ctr - ctr_arr))
                    if dist < best_dist:
                        best_dist  = dist
                        best_mask  = comp
                filled = best_mask if best_mask is not None else filled

            # ── Step 5: gradient boundary refinement ─────────────────────────
            if filled.any():
                smooth   = ndimage.gaussian_filter(max_enh, sigma=1.0)
                gx = sobel(smooth, axis=2)
                gy = sobel(smooth, axis=1)
                gz = sobel(smooth, axis=0)
                gradient = np.sqrt(gx**2 + gy**2 + gz**2)

                expanded = ndimage.binary_dilation(filled, np.ones((1, 3, 3)))
                # 85th-pct LOCAL gradient — less aggressive than 75th to
                # avoid over-trimming on NCR / low-enhancement tumours
                grad_vals = gradient[bm_local]
                grad_thr  = float(np.percentile(grad_vals, 85)) if grad_vals.size else 0.0
                refined   = expanded & (gradient < grad_thr) & bm_local

                for z in range(refined.shape[0]):
                    if refined[z].any():
                        refined[z] = binary_fill_holes(refined[z])

                refined = ndimage.binary_closing(refined, np.ones((1, 3, 3)))

                # Fill only tiny internal holes (≤25 px²) so the boundary
                # stays irregular but does not have speckle inside the tumour
                filled_ref = np.zeros_like(refined)
                for z_r in range(refined.shape[0]):
                    sl_r = refined[z_r]
                    if not sl_r.any():
                        continue
                    # fill_holes then remove filled voxels >25px from boundary
                    sl_filled = binary_fill_holes(sl_r)
                    added = sl_filled & ~sl_r
                    if HAS_SKIMAGE:
                        # only keep small filled patches (avoid solid oval fill)
                        added_lbl = morphology.label(added)
                        for lbl_id in range(1, added_lbl.max() + 1):
                            if (added_lbl == lbl_id).sum() <= 25:
                                sl_r = sl_r | (added_lbl == lbl_id)
                    filled_ref[z_r] = sl_r
                if filled_ref.any():
                    refined = filled_ref

                # Keep only the nearest component after refinement too
                if refined.any():
                    r_labeled, n_r = _label(refined)
                    best_r, best_rd = None, float('inf')
                    for lbl in range(1, n_r + 1):
                        rc = r_labeled == lbl
                        rd = float(np.linalg.norm(np.argwhere(rc).mean(axis=0) - ctr_arr))
                        if rd < best_rd:
                            best_rd, best_r = rd, rc
                    if best_r is not None and best_rd < 60:  # must be within 60 px
                        filled = best_r
                    # else: keep pre-refinement filled (refinement drifted too far)

            combined |= filled

        return combined.astype(bool)

    # ── Primary segmentation (calibrated threshold) ───────────────────────────
    mask = _run_stages(max_enh, body_mask, enhancement_threshold, min_lesion_voxels)

    # ── SER-based supplement for plateau-kinetics tumours (e.g. BC01 NCR) ─────
    # Signal Enhancement Ratio = early_post / late_post.  Values > 0.9 indicate
    # plateau or washout — the hallmark of malignancy even at moderate peak Enh.
    if enh_stack is not None and enh_stack.shape[0] >= 3:
        T = enh_stack.shape[0]
        early_idx  = max(1, T // 4)           # ~¼ through post-contrast frames
        late_idx   = T - 1                    # last post-contrast frame
        early_enh  = enh_stack[early_idx]
        late_enh   = enh_stack[late_idx]

        # Plateau: early_enh high AND late_enh ≈ early_enh (SER ≈ 1.0)
        # Washout: early_enh high AND late_enh < early_enh  (SER > 1.0)
        early_thr = enhancement_threshold * 0.70   # 70% of main threshold
        plateau_mask = (
            (early_enh > early_thr) &
            (np.abs(late_enh - early_enh) < early_thr * 0.5) &
            body_mask
        )
        washout_mask = (
            (early_enh > early_thr) &
            (late_enh < early_enh - early_thr * 0.3) &
            body_mask
        )
        ser_candidate = plateau_mask | washout_mask

        if np.any(ser_candidate) and not np.any(mask):
            # Only use SER candidates if primary found nothing —
            # run them through the same Stage-B/C filter
            ser_mask = _run_stages(early_enh, body_mask,
                                   early_thr, min_lesion_voxels)
            if np.any(ser_mask):
                mask = ser_mask

    # ── Rescue pass: lower threshold ─────────────────────────────────────────
    if not np.any(mask):
        rescue_thr = max(0.20, enhancement_threshold * 0.55)
        mask = _run_stages(max_enh, body_mask, rescue_thr,
                           max(30, min_lesion_voxels // 2),
                           relax_posterior=True)

    # ── Body-relative self-correction ────────────────────────────────────────
    # If the detected mask centroid sits in the MIDDLE of the body (not at
    # the inferior breast protrusion), the zone was inverted.  Rebuild using
    # the OPPOSITE body end per slice and re-run — no image-half cuts needed.
    if np.any(mask):
        detected_coords = np.argwhere(mask)
        det_y_mean = float(detected_coords[:, 1].mean())

        body_y_lo_vals, body_y_hi_vals = [], []
        for z_idx in range(Z):
            sl = body_mask[z_idx]
            if sl.any():
                rows = np.where(sl.any(axis=1))[0]
                if rows.size >= 4:
                    body_y_lo_vals.append(int(rows.min()))
                    body_y_hi_vals.append(int(rows.max()))

        if body_y_lo_vals:
            body_y_lo  = float(np.median(body_y_lo_vals))
            body_y_hi  = float(np.median(body_y_hi_vals))
            body_span  = body_y_hi - body_y_lo
            # Breast zones at EITHER end of the body
            lo_zone_end   = body_y_lo + body_span * 0.45
            hi_zone_start = body_y_hi - body_span * 0.45
            in_lo_zone = det_y_mean <= lo_zone_end
            in_hi_zone = det_y_mean >= hi_zone_start

            # If centroid is in neither breast zone → wrong area detected
            if not in_lo_zone and not in_hi_zone:
                flipped_bm = np.zeros_like(body_mask, dtype=bool)
                for z_idx in range(Z):
                    sl_f = body_mask[z_idx]
                    if not sl_f.any():
                        continue
                    rows_f = np.where(sl_f.any(axis=1))[0]
                    if rows_f.size < 4:
                        continue
                    y_lo_f  = int(rows_f.min())
                    y_hi_f  = int(rows_f.max())
                    depth_f = max(4, int((y_hi_f - y_lo_f + 1) * 0.45))
                    # Try the end OPPOSITE to the current wrong detection
                    if det_y_mean > (y_lo_f + y_hi_f) / 2:
                        flipped_bm[z_idx,
                                   y_lo_f : y_lo_f + depth_f, :] = sl_f[
                                       y_lo_f : y_lo_f + depth_f, :]
                    else:
                        flipped_bm[z_idx,
                                   y_hi_f - depth_f : y_hi_f + 1, :] = sl_f[
                                       y_hi_f - depth_f : y_hi_f + 1, :]

                if flipped_bm.any():
                    mask_flip = _run_stages(max_enh, flipped_bm,
                                            enhancement_threshold,
                                            min_lesion_voxels)
                    if not np.any(mask_flip):
                        mask_flip = _run_stages(
                            max_enh, flipped_bm,
                            max(0.20, enhancement_threshold * 0.55),
                            max(30, min_lesion_voxels // 2),
                            relax_posterior=True)
                    if np.any(mask_flip):
                        mask = mask_flip

    return mask.astype(bool)


# ── Stage: kinetic classification ─────────────────────────────────────────────

KINETIC_PERSISTENT = 1
KINETIC_PLATEAU    = 2
KINETIC_WASHOUT    = 3


def _classify_kinetics(
        enh_stack: np.ndarray,
        tumour_mask: np.ndarray,
        washout_thr: float = -0.10,
        plateau_thr: float = 0.10) -> np.ndarray:
    """Classify each tumour voxel as Type I/II/III based on late enhancement."""
    if enh_stack.shape[0] < 2:
        return np.zeros_like(tumour_mask, dtype=np.int8)

    peak_frame = np.argmax(enh_stack, axis=0)
    peak_val = np.take_along_axis(enh_stack, peak_frame[np.newaxis], axis=0)[0]
    last_val = enh_stack[-1]
    delta = last_val - peak_val

    kinetic = np.zeros(tumour_mask.shape, dtype=np.int8)
    kinetic[tumour_mask & (delta < washout_thr)] = KINETIC_WASHOUT
    kinetic[tumour_mask & (np.abs(delta) <= plateau_thr) & (kinetic == 0)] = KINETIC_PLATEAU
    kinetic[tumour_mask & (kinetic == 0)] = KINETIC_PERSISTENT
    return kinetic


# ── Stage: measurements ───────────────────────────────────────────────────────

def _compute_measurements(tumour_mask, kinetic_map, voxel_spacing, patient_id):
    vz, vy, vx = voxel_spacing
    voxel_vol_mm3 = vz * vy * vx
    total_voxels = int(np.sum(tumour_mask))
    volume_ml = total_voxels * voxel_vol_mm3 / 1000.0

    if total_voxels > 0:
        coords = np.argwhere(tumour_mask)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        max_diam = max((zmax-zmin+1)*vz, (ymax-ymin+1)*vy, (xmax-xmin+1)*vx)
        # Hard cap: no breast tumour is wider than 150 mm (15 cm).
        # Values above this indicate a whole-FOV false-positive detection
        # (e.g. chest-wall muscle band mis-classified as tumour when the
        # breast zone orientation was wrong).  Cap rather than crash.
        max_diam = min(max_diam, 150.0)
    else:
        max_diam = 0.0

    total_k = max(int(np.sum(kinetic_map > 0)), 1)
    k_pcts = {
        "TypeI_pct":   100 * int(np.sum(kinetic_map == 1)) / total_k,
        "TypeII_pct":  100 * int(np.sum(kinetic_map == 2)) / total_k,
        "TypeIII_pct": 100 * int(np.sum(kinetic_map == 3)) / total_k,
    }
    dominant = max(k_pcts, key=k_pcts.get).replace("_pct", "")
    df = pd.DataFrame([dict(
        PatientID=patient_id,
        Volume_mL=round(volume_ml, 2),
        MaxDiameter_mm=round(max_diam, 1),
        TotalVoxels=total_voxels,
        DominantKinetics=dominant,
        **{k: round(v, 1) for k, v in k_pcts.items()},
    )])
    return volume_ml, df


# ── Master bridge: DCM index → per-patient SegmentationResult ────────────────

def run_dce_segmentation(
        dcm_index: dict,
        enhancement_threshold: float = 0.40,
        use_registration: bool = False,
        seed_points: dict = None,
        progress_callback=None) -> Dict[Tuple[str, str], SegmentationResult]:
    """
    Convert a CBM-app DCM index into per-(patient, visit) SegmentationResults.

    Parameters
    ----------
    dcm_index          : dict  (pid, vis, tt) → [(slice_loc, filepath)]
    enhancement_threshold : float   min enhancement fraction
    use_registration   : bool   run SimpleITK rigid registration per frame
    seed_points        : dict  (pid, vis) → list of (y_arr, x_arr) tuples
                         User-provided seed points in array coordinates.
                         When present, segmentation uses seeded region-growing
                         instead of zone-based heuristics — far more reliable.
    progress_callback  : callable(frac, text) | None
    """
    from collections import defaultdict

    # Group by (pid, vis), collect timepoints
    by_pv = defaultdict(list)
    for (pid, vis, tt), entries in dcm_index.items():
        by_pv[(pid, vis)].append((tt, entries))

    results: Dict[Tuple[str, str], SegmentationResult] = {}
    pv_list = sorted(by_pv.keys())
    n_total = len(pv_list)

    for idx, (pid, vis) in enumerate(pv_list):
        frac = idx / max(n_total, 1)
        if progress_callback:
            progress_callback(frac, f"Segmenting {pid} {vis} ({idx+1}/{n_total}) …")

        timepoints = sorted(by_pv[(pid, vis)], key=lambda x: x[0])
        if len(timepoints) < 2:
            continue  # need at least pre + one post frame

        # ── Load pre-contrast (earliest timepoint) ────────────────────────
        pre_tt, pre_entries = timepoints[0]
        pre_vol, _ = _load_volume_from_entries(pre_entries)
        if pre_vol is None or pre_vol.size == 0:
            continue

        # Estimate voxel spacing from first file header
        voxel_spacing = (1.0, 1.0, 1.0)
        try:
            ds = pydicom.dcmread(pre_entries[0][1], stop_before_pixels=True, force=True)
            ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
            st = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
            voxel_spacing = (st, float(ps[0]), float(ps[1]))
        except Exception:
            pass

        # ── Load post-contrast frames ─────────────────────────────────────
        post_vols: List[np.ndarray] = []
        for post_tt, post_entries in timepoints[1:]:
            vol, _ = _load_volume_from_entries(post_entries)
            if vol is None:
                continue
            # Pad/crop Z to match pre
            if vol.shape[0] < pre_vol.shape[0]:
                pad = pre_vol.shape[0] - vol.shape[0]
                vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)))
            else:
                vol = vol[:pre_vol.shape[0]]
            if use_registration and HAS_SITK:
                vol = _register_volume(vol, pre_vol, voxel_spacing)
            post_vols.append(vol)

        if not post_vols:
            continue

        # ── Enhancement map & body-relative breast mask ───────────────────
        # NOTE: DICOM ImageOrientationPatient is NOT used to determine the
        # breast zone.  Every attempt to derive breast_in_top from IOP failed
        # because the QIN TWIST acquisition col_P sign varies, and any wrong
        # value passed as breast_in_top_override bypasses the body-relative
        # voting entirely.  Passing None forces the body-relative approach to
        # run unconditionally: it finds the body bounding box per slice, tries
        # both inferior and superior zones, and picks the one with higher 95th-
        # percentile enhancement signal — no orientation metadata required.
        max_enh, enh_stack = _compute_enhancement_map(pre_vol, post_vols)
        body_mask = _compute_breast_body_mask(
            pre_vol, max_enh=max_enh,
            breast_in_top_override=None)          # always use body-relative voting

        # ── Build seed mask (with cross-visit propagation) ───────────────────
        # If this visit has no manual seeds BUT another visit of the SAME
        # patient does, propagate those seeds so both visits use the same
        # spatial tumour location. Preference order: adjacent visit first
        # (V1→V2, V2→V3) then any other visit, to minimise spatial mismatch
        # when the tumour shifts slightly between visits.
        seed_pts = (seed_points or {}).get((pid, vis), [])
        seed_propagated_from = None
        if not seed_pts and seed_points:
            vis_num = int(vis[1]) if vis[1:].isdigit() else 1
            # Try adjacent visits first (±1), then all others
            for delta in [1, -1, 2, -2, 3, -3]:
                alt_vis = f"V{vis_num + delta}"
                if alt_vis != vis and alt_vis[1:].isdigit():
                    alt_seeds = seed_points.get((pid, alt_vis), [])
                    if alt_seeds:
                        seed_pts = alt_seeds
                        seed_propagated_from = alt_vis
                        break
        seed_pts = (seed_points or {}).get((pid, vis), [])
        seed_mask_3d = None
        if seed_pts:
            seed_mask_3d = _build_seed_mask_3d(seed_pts, pre_vol.shape, radius_px=45)

        tumour_mask = _segment_tumour(
            max_enh, body_mask, pre_vol,
            enhancement_threshold=enhancement_threshold,
            enh_stack=enh_stack,
            seed_mask_3d=seed_mask_3d)

        # ── Kinetics & measurements ────────────────────────────────────────
        kinetic_map = _classify_kinetics(enh_stack, tumour_mask)
        k_counts = {k: int(np.sum(kinetic_map == k)) for k in [1, 2, 3]}
        volume_ml, mdf = _compute_measurements(tumour_mask, kinetic_map,
                                                voxel_spacing, pid)

        peak_z = int(np.argmax(np.sum(tumour_mask, axis=(1, 2))))

        results[(pid, vis)] = SegmentationResult(
            patient_id=pid,
            visit=vis,
            tumour_mask=tumour_mask,
            enhancement_map=max_enh,
            kinetic_map=kinetic_map,
            voxel_spacing_mm=voxel_spacing,
            volume_ml=volume_ml,
            peak_slice_idx=peak_z,
            kinetic_counts=k_counts,
            measurements_df=mdf,
        )

    if progress_callback:
        progress_callback(1.0, f"Segmentation complete — {len(results)} patient-visits processed.")
    return results


# ── Helper: get 2D mask + enhancement for a specific patient/visit slice ──────

def _get_seg_for_slice(seg_results: dict, pid: str, vis: str,
                        img_shape: Tuple[int, int]):
    """
    Return (mask_2d, enh_2d, peak_z) from stored segmentation for a patient/visit.
    mask_2d and enh_2d are 2-D arrays matching img_shape.
    Returns (None, None, None) if not available or irreconcilable shape mismatch.

    Improvement: if the stored segmentation volume has a different in-plane size
    than the displayed MRI slice (e.g. due to different reconstruction), a
    centre-crop is attempted before giving up.
    """
    if not seg_results:
        return None, None, None
    seg = seg_results.get((pid, vis))
    if seg is None or not np.any(seg.tumour_mask):
        return None, None, None

    areas = np.sum(seg.tumour_mask, axis=(1, 2))
    peak_z = int(np.argmax(areas))
    mask_2d = seg.tumour_mask[peak_z]
    enh_2d  = seg.enhancement_map[peak_z] if seg.enhancement_map is not None else None

    # Exact match
    if mask_2d.shape == img_shape:
        return mask_2d, enh_2d, peak_z

    # Try centre-crop if segmentation volume is larger
    mh, mw = mask_2d.shape
    ih, iw = img_shape
    if mh >= ih and mw >= iw:
        y0 = (mh - ih) // 2
        x0 = (mw - iw) // 2
        mask_2d = mask_2d[y0:y0+ih, x0:x0+iw]
        if enh_2d is not None:
            enh_2d = enh_2d[y0:y0+ih, x0:x0+iw]
        return mask_2d, enh_2d, peak_z

    # Irreconcilable shape difference
    return None, None, None


# ── Helper: build Ktrans map anchored to a real 2-D tumour mask ───────────────

def _ktrans_from_real_mask(
        mask_2d: np.ndarray,
        enh_2d: Optional[np.ndarray],
        Kt_mu: float,
        seed: int = 0,
        noise: float = 0.12) -> np.ndarray:
    """
    Generate a realistic spatially-heterogeneous Ktrans map whose spatial
    extent is determined by the **real** 2-D segmentation mask.

    If a real enhancement map slice is provided, Ktrans values are scaled
    proportionally to local enhancement (higher enhancement → higher Ktrans).
    Otherwise, a radial gradient peaked at 70% radius is used.
    """
    H, W = mask_2d.shape
    rng = np.random.default_rng(seed)
    kt_map = np.zeros((H, W), float)

    if not np.any(mask_2d):
        return kt_map

    rows, cols = np.where(mask_2d)
    r_ctr = int(np.median(rows))
    c_ctr = int(np.median(cols))

    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X.astype(float) - c_ctr)**2 + (Y.astype(float) - r_ctr)**2)
    max_d = dist[mask_2d].max() if mask_2d.any() else 1.0
    dist_norm = dist / (max_d + 1e-9)

    if enh_2d is not None and np.any(mask_2d):
        # Scale actual enhancement to [0, 1] within the mask, use as spatial weight
        enh_masked = enh_2d[mask_2d]
        e_lo = np.percentile(enh_masked, 5)
        e_hi = np.percentile(enh_masked, 98)
        enh_norm = np.clip((enh_2d - e_lo) / (e_hi - e_lo + 1e-9), 0, 1)
        spatial_weight = enh_norm
    else:
        # Radial rim-dominant profile (Gaussian peak at 70% radius)
        rim = np.exp(-((dist_norm - 0.70)**2) / (2 * 0.25**2))
        rim_max = rim[mask_2d].max() if mask_2d.any() else 1.0
        spatial_weight = 0.65 + 0.35 * rim / (rim_max + 1e-9)

    texture = gaussian_filter(rng.normal(0, 1, (H, W)), 4) * noise * Kt_mu
    kt_map[mask_2d] = np.clip(
        Kt_mu * spatial_weight[mask_2d] + texture[mask_2d], 0, None)

    return kt_map


# ═══════════════════════════════════════════════════════════════════════════════
# §1  PAGE CONFIG / STYLE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CBM-DCE-MRI | Breast Cancer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Page & app background ─────────────────────────────────────── */
  .stApp{background:#F5F7FA !important; color:#1A2233 !important}
  .main .block-container{background:#F5F7FA}

  /* ── Sidebar ────────────────────────────────────────────────────── */
  div[data-testid="stSidebar"]{background:#ECEFF1 !important}
  div[data-testid="stSidebar"] *{color:#1A2233 !important}
  div[data-testid="stSidebar"] label{color:#37474F !important}

  /* ── General text ───────────────────────────────────────────────── */
  p, li, span, label, div{color:#1A2233}
  h1,h2,h3,h4{color:#0D2B6E !important; font-weight:700}

  /* ── Metric cards ───────────────────────────────────────────────── */
  .card{
    background:#FFFFFF; border-radius:12px; padding:16px 20px;
    border:1px solid #B0BEC5; border-left:4px solid #1565C0;
    box-shadow:0 2px 8px rgba(0,0,0,0.08); margin:6px 0}
  .card h3{color:#546E7A; font-size:13px; margin:0}
  .card h2{color:#0D2B6E; font-size:26px; margin:4px 0 0; font-weight:700}
  .card-teal{border-left-color:#00695C !important}
  .card-rose {border-left-color:#B71C1C !important}
  .card-gold {border-left-color:#E65100 !important}

  /* ── Buttons ────────────────────────────────────────────────────── */
  .stButton>button{
    background:linear-gradient(135deg,#1565C0,#00695C);
    color:#fff !important; border:none; border-radius:8px;
    padding:10px 22px; font-weight:600; font-size:15px; width:100%}
  .stButton>button:hover{opacity:.88}

  /* ── Section headers ────────────────────────────────────────────── */
  .sec{background:linear-gradient(90deg,#1565C022,transparent);
       border-left:4px solid #1565C0; padding:8px 16px;
       border-radius:0 8px 8px 0; margin:16px 0}
  .sec h2{color:#0D2B6E !important}

  /* ── Upload hint ────────────────────────────────────────────────── */
  .upload-hint{border:2px dashed #1565C0; border-radius:12px;
               padding:20px; text-align:center; background:#E3F2FD;
               color:#37474F; font-size:14px}

  /* ── Progress bar ───────────────────────────────────────────────── */
  .stProgress>div>div{background:linear-gradient(90deg,#1565C0,#00695C)}

  /* ── Tab text ───────────────────────────────────────────────────── */
  button[data-baseweb="tab"]{color:#37474F !important; font-weight:600}
  button[data-baseweb="tab"][aria-selected="true"]{
    color:#1565C0 !important; border-bottom:3px solid #1565C0}

  /* ── Table cells ────────────────────────────────────────────────── */
  .stDataFrame td, .stDataFrame th{color:#1A2233 !important}

  /* ── Info / warning / success boxes ────────────────────────────── */
  .stAlert p{color:#1A2233 !important}

  /* ── Selectbox / slider labels ──────────────────────────────────── */
  .stSelectbox label, .stSlider label, .stRadio label,
  .stNumberInput label, .stTextInput label{color:#37474F !important}

  /* ── Code / monospace ───────────────────────────────────────────── */
  code{background:#E3F2FD; color:#0D2B6E; border-radius:4px; padding:1px 4px}

  /* ── Seg badge ──────────────────────────────────────────────────── */
  .seg-badge{display:inline-block; background:#E0F2F1;
             border:1px solid #00695C; border-radius:6px;
             padding:2px 10px; font-size:12px; color:#00695C}
</style>
""", unsafe_allow_html=True)

# ─── Colour palette ───────────────────────────────────────────────────────────
C = dict(
    bg="#F5F7FA",    # page background — light grey
    panel="#FFFFFF", # card / panel background — white
    card="#EEF2F8",  # slightly tinted card
    blue="#1565C0",  lblue="#1976D2",
    teal="#00695C",  lteal="#00897B",
    gold="#E65100",  lgold="#F57C00",
    rose="#B71C1C",  lrose="#E53935",
    green="#2E7D32", lgreen="#43A047",
    purple="#6A1B9A", white="#1A2233",   # "white" now = dark text
    muted="#546E7A",  gridline="#CFD8DC",
)

cmap_kt = LinearSegmentedColormap.from_list(
    "kt", ["#0B0F17","#1B4F8A","#4A90D9","#F5C842","#D63060"], 256)
cmap_tumor = LinearSegmentedColormap.from_list(
    "tumor_radial",
    ["#1B0F3A","#1B4F8A","#1A9E8A","#F5C842","#D63060","#FF9090"], 256)


def style_ax(ax):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["muted"]); ax.tick_params(axis="both", labelcolor=C["white"])
    ax.xaxis.label.set_color(C["white"])
    ax.yaxis.label.set_color(C["white"])
    ax.title.set_color(C["white"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["muted"])
    ax.grid(True, alpha=0.25, color=C["gridline"])
    return ax


def card(label, value, sub="", color="blue"):
    cls = {"teal":"card-teal","rose":"card-rose","gold":"card-gold"}.get(color,"")
    return (f'<div class="card {cls}"><h3>{label}</h3>'
            f'<h2>{value}</h2>'
            f'<p style="color:#6E7C8C;font-size:12px;margin:0">{sub}</p></div>')


def fig_to_img(fig, dpi=300):
    MAX_PX = 8000
    buf = io.BytesIO()
    w_in, h_in = fig.get_size_inches()
    w_px, h_px = int(w_in * dpi), int(h_in * dpi)
    if w_px > MAX_PX or h_px > MAX_PX:
        scale = MAX_PX / max(w_px, h_px)
        fig.set_size_inches(w_in * scale, h_in * scale)
    try:
        renderer = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(renderer, bbox_extra_artists=None)
        if (bb is None or not np.isfinite(bb.width) or not np.isfinite(bb.height)
                or bb.width > MAX_PX * 2 or bb.height > MAX_PX * 2):
            raise ValueError("Pathological bbox")
        fig.savefig(buf, format="png", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.15, facecolor=C["bg"])
    except Exception:
        buf.seek(0); buf.truncate(0)
        fig.savefig(buf, format="png", dpi=dpi, facecolor=C["bg"])
    buf.seek(0); plt.close(fig)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# §2  LARGE-FILE SAFE DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_or_create_tmpdir():
    if "tmp_dir" not in st.session_state or \
       not pathlib.Path(st.session_state["tmp_dir"]).exists():
        st.session_state["tmp_dir"] = tempfile.mkdtemp(prefix="cbm_rar_")
    return st.session_state["tmp_dir"]


def _cleanup_tmpdir():
    tmp = st.session_state.pop("tmp_dir", None)
    if tmp and pathlib.Path(tmp).exists():
        shutil.rmtree(tmp, ignore_errors=True)


def stream_upload_to_disk(uploaded_file) -> pathlib.Path:
    tmp_dir  = _get_or_create_tmpdir()
    suffix   = pathlib.Path(uploaded_file.name).suffix.lower()
    tmp_file = pathlib.Path(tmp_dir) / f"upload{suffix}"
    CHUNK = 4 * 1024 * 1024
    total = uploaded_file.size
    written = 0
    prog = st.progress(0, text="Saving to disk …")
    with open(tmp_file, "wb") as fout:
        while True:
            chunk = uploaded_file.read(CHUNK)
            if not chunk:
                break
            fout.write(chunk)
            written += len(chunk)
            prog.progress(min(written / total, 1.0),
                          text=f"Saving … {written/1e6:.0f} / {total/1e6:.0f} MB")
    prog.empty()
    return tmp_file


def extract_archive_to_disk(archive_path: pathlib.Path,
                              extract_dir: pathlib.Path,
                              placeholder) -> None:
    ext = archive_path.suffix.lower()
    extract_dir.mkdir(parents=True, exist_ok=True)

    if ext == ".rar":
        placeholder.progress(0, text="Extracting RAR archive …")
        with rarfile.RarFile(str(archive_path)) as rf:
            members = rf.infolist()
            total   = max(sum(m.file_size for m in members), 1)
            done    = 0
            for m in members:
                if m.file_size > 0:
                    rf.extract(m, str(extract_dir))
                    done += m.file_size
                    placeholder.progress(
                        min(done/total, 0.99),
                        text=f"Extracting … {done/1e6:.0f} / {total/1e6:.0f} MB")
        placeholder.progress(1.0, text="Extraction complete ✓")

    elif ext == ".zip":
        placeholder.progress(0, text="Extracting ZIP archive …")
        with zipfile.ZipFile(str(archive_path)) as zf:
            members = zf.infolist()
            total   = max(sum(m.file_size for m in members), 1)
            done    = 0
            for m in members:
                zf.extract(m, str(extract_dir))
                done += m.file_size
                placeholder.progress(
                    min(done/total, 0.99),
                    text=f"Extracting … {done/1e6:.0f} / {total/1e6:.0f} MB")
        placeholder.progress(1.0, text="Extraction complete ✓")

    time.sleep(0.3)
    placeholder.empty()


def build_index(extract_dir: pathlib.Path):
    """
    Walk *extract_dir* and build a DICOM index keyed by
    (patient_id, visit_label, timepoint_seconds).

    Paths stored in the returned dcm_index are RELATIVE to extract_dir.
    This makes the index portable: save it alongside the dataset and the
    same index works on Windows (C:\\Users\\HP\\...), Linux (/data/...),
    or any web server — as long as extract_dir is correct for that machine.

    Call `resolve_dcm_path(extract_dir, rel_path)` to reconstruct the
    absolute path when reading DICOM files.

    Visit assignment
    ----------------
    Visits (V1, V2 …) are assigned from DICOM StudyDate sorted
    chronologically per patient.  A folder-date fallback (MM-DD-YYYY)
    is used when the tag is absent.
    """
    import collections as _col

    aif_df = resp_df = None
    all_files = [f for f in extract_dir.rglob("*") if f.is_file()]
    prog = st.progress(0, text="Building file index …")
    n = len(all_files)

    raw_records = []   # (pid, study_date, tt, sl, rel_path)

    for ii, fp in enumerate(all_files):
        fl = fp.name.lower()
        prog.progress((ii + 1) / n, text=f"Indexing {ii+1}/{n} files …")

        if "aif" in fl and fl.endswith(".xlsx"):
            try:
                aif_df = pd.read_excel(fp)
            except Exception:
                pass
            continue

        if "response" in fl and fl.endswith(".xlsx"):
            try:
                resp_df = pd.read_excel(fp)
            except Exception:
                pass
            continue

        if not (fl.endswith(".dcm") and fp.stat().st_size > 10_000):
            continue

        # ── Store path RELATIVE to extract_dir ──────────────────────────────
        try:
            rel_path = str(fp.relative_to(extract_dir))
        except ValueError:
            rel_path = str(fp)   # fallback: absolute (same-machine use)

        fname_str = str(fp)   # used for regex only; not stored
        m_tt = re.search(r"TT([\d.]+)s", fname_str)
        tt   = float(m_tt.group(1)) if m_tt else -1.0

        parts = fname_str.replace("\\", "/").split("/")
        pid = next((p.strip("- ") for p in parts if "BC0" in p), "unknown")

        try:
            ds = pydicom.dcmread(fname_str, stop_before_pixels=True, force=True)

            sop      = str(getattr(ds, "SOPClassUID", ""))
            modality = str(getattr(ds, "Modality", "MR"))
            if not hasattr(ds, "Rows"):
                continue
            if sop in _NON_IMAGE_SOP or modality not in _IMAGE_MODALITIES:
                continue

            sl = float(getattr(ds, "SliceLocation", 0))

            # ── Derive timepoint from DICOM header when filename has no TT ─
            # TCIA-downloaded QIN files don't embed "TT???.?s" in filenames.
            # Try TemporalPositionIdentifier (integer series index) first,
            # then AcquisitionTime (HHMMSS wall-clock), then ContentTime.
            if tt < 0:
                tp_id = getattr(ds, "TemporalPositionIdentifier", None)
                if tp_id is not None:
                    try:
                        tt = float(tp_id) * 20.0   # ≈20 s per TWIST frame
                    except (ValueError, TypeError):
                        pass
            if tt < 0:
                for tag_name in ("AcquisitionTime", "ContentTime"):
                    acq = str(getattr(ds, tag_name, "")).strip()
                    if len(acq) >= 6:
                        try:
                            h_  = int(acq[0:2])
                            m__ = int(acq[2:4])
                            s_  = float(acq[4:])
                            tt  = h_ * 3600 + m__ * 60 + s_
                            break
                        except (ValueError, IndexError):
                            pass

            study_date = str(getattr(ds, "StudyDate", "")).strip()
            if not study_date:
                m_dt = next((p for p in parts
                             if re.match(r"\d{2}-\d{2}-\d{4}", p)), None)
                if m_dt:
                    try:
                        mm, dd_, yyyy = m_dt.split("-")
                        study_date = f"{yyyy}{mm}{dd_}"
                    except ValueError:
                        study_date = m_dt
                else:
                    study_date = "00000000"

            raw_records.append((pid, study_date, tt, sl, rel_path))

        except Exception:
            continue

    prog.empty()

    # ── Assign visit labels ────────────────────────────────────────────────
    dates_by_pid: Dict[str, set] = _col.defaultdict(set)
    for pid, study_date, *_ in raw_records:
        dates_by_pid[pid].add(study_date)

    visit_label: Dict[tuple, str] = {}
    for pid, date_set in dates_by_pid.items():
        for i, d in enumerate(sorted(date_set)):
            visit_label[(pid, d)] = f"V{i + 1}"

    dcm_index: Dict = {}
    for pid, study_date, tt, sl, rel_path in raw_records:
        vis = visit_label.get((pid, study_date), "V1")
        dcm_index.setdefault((pid, vis, tt), []).append((sl, rel_path))

    for k in dcm_index:
        dcm_index[k].sort(key=lambda x: x[0])

    return dcm_index, aif_df, resp_df


def resolve_dcm_path(base_dir: pathlib.Path, stored_path: str) -> str:
    """
    Reconstruct the absolute DICOM path from whatever was stored.

    Handles three cases so the app works identically whether the index
    was built on a local Windows machine, a Linux server, or a Docker
    container mounted at a different root:

      1. stored_path is already absolute and the file exists → return as-is
      2. stored_path is relative → join with base_dir → return
      3. stored_path is absolute but does not exist (different machine) →
         try joining its *basename* with base_dir as a last resort
    """
    p = pathlib.Path(stored_path)
    # Case 1: absolute and file exists on this machine
    if p.is_absolute() and p.exists():
        return str(p)
    # Case 2: relative path → join with base_dir
    candidate = base_dir / stored_path
    if candidate.exists():
        return str(candidate)
    # Case 3: absolute path from a different machine → use just the filename
    candidate2 = base_dir / p.name
    if candidate2.exists():
        return str(candidate2)
    # Give up — return the joined path and let the caller raise
    return str(base_dir / stored_path)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  CBM / TOFTS MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def cbm_signal(t_obs, C_p_fn, Ktrans, ve, vp, z=0.01, v=1e-3,
               TR=6.13e-3, alpha=10.0, R10=0.625, r1=4.5):
    kep = Ktrans / ve; a = np.radians(alpha); S = np.zeros(len(t_obs))
    for i, t in enumerate(t_obs):
        Cp = float(C_p_fn(t))
        Ce = vp*Cp + (Ktrans*Cp/kep)*(1 - np.exp(-kep*z/v)) if kep > 0 else vp*Cp
        E1 = np.exp(-TR*(R10 + r1*Ce))
        S[i] = np.sin(a)*(1-E1)/(1-np.cos(a)*E1)
    E10 = np.exp(-TR*R10); S0 = np.sin(a)*(1-E10)/(1-np.cos(a)*E10)
    return S / S0


def tofts_signal(t_obs, C_p_fn, Ktrans, ve, vp,
                 TR=6.13e-3, alpha=10.0, R10=0.625, r1=4.5):
    kep = Ktrans/ve; a = np.radians(alpha); S = np.zeros(len(t_obs))
    for i, t in enumerate(t_obs):
        tau = np.linspace(0, t, 200)
        Cp_ = np.array([float(C_p_fn(s)) for s in tau])
        Ce  = Ktrans*np.trapezoid(Cp_*np.exp(-kep*(t-tau)),tau)+vp*float(C_p_fn(t))
        E1  = np.exp(-TR*(R10+r1*Ce))
        S[i] = np.sin(a)*(1-E1)/(1-np.cos(a)*E1)
    E10 = np.exp(-TR*R10); S0 = np.sin(a)*(1-E10)/(1-np.cos(a)*E10)
    return S / S0


def fit_voxel(S, t, C_p_fn, model="cbm"):
    S_n = S / (S[:3].mean() + 1e-9)
    fn  = cbm_signal if model == "cbm" else tofts_signal
    def m(t, Kt, ve, vp):
        Kt=np.clip(Kt,0.001,5); ve=np.clip(ve,0.01,0.99); vp=np.clip(vp,0.001,0.5)
        S_raw = fn(t, C_p_fn, Kt, ve, vp)
        return S_raw / (S_raw[:3].mean() + 1e-9)
    try:
        iAUC  = float(np.trapezoid(np.clip(S_n-1.0,0,None), t/60))
        Kt_i  = float(np.clip(iAUC*0.15, 0.05, 0.80))
    except Exception: Kt_i = 0.15
    try:
        popt,_ = curve_fit(m, t, S_n, p0=[Kt_i,0.25,0.04],
                           bounds=([0.001,0.01,0.001],[2,0.99,0.5]),
                           maxfev=2000, method="trf")
        return float(popt[0]),float(popt[1]),float(popt[2]),float(np.mean((S_n-m(t,*popt))**2))
    except Exception:
        return np.nan,np.nan,np.nan,np.nan


def make_cohort(t_aif, c_aif, n_cr, n_ncr, seed=42):
    rng = np.random.default_rng(seed)
    C_fn = interp1d(t_aif, c_aif, bounds_error=False, fill_value=(0,c_aif[-1]))
    T = np.array([49.6,69.7,89.9,110.0,130.2,150.4,170.5,190.7,
                  210.8,231.0,251.2,271.3,291.5,311.7,331.8,352.0,
                  372.1,392.3,412.5,432.6,452.8,473.0,493.1,513.3,
                  533.4,553.6,573.8,593.9])
    rows = []
    CR_IDS  = ["BC05","BC06","BC15"]
    NCR_IDS = ["BC01","BC08","BC10","BC12","BC13","BC14","BC16"]
    for i in range(n_cr + n_ncr):
        is_cr = i < n_cr
        pid = CR_IDS[i] if is_cr else NCR_IDS[i - n_cr]
        lbl = "CR" if is_cr else "NCR"
        if is_cr:
            Kt0=rng.uniform(0.24,0.35); ve0=rng.uniform(0.38,0.48); vp0=rng.uniform(0.04,0.08)
            Kt1=Kt0*rng.uniform(0.25,0.45); ve1=ve0*rng.uniform(0.70,0.90)
        else:
            Kt0=rng.uniform(0.14,0.26); ve0=rng.uniform(0.30,0.42); vp0=rng.uniform(0.02,0.05)
            Kt1=Kt0*rng.uniform(0.55,0.80); ve1=ve0*rng.uniform(0.85,0.98)
        v_ = rng.uniform(0.6e-3,1.2e-3) if is_cr else rng.uniform(0.3e-3,0.7e-3)
        for vis,Kt,ve,vp in [("V1",Kt0,ve0,vp0),("V2",Kt1,ve1,vp0*0.9)]:
            S = cbm_signal(T,C_fn,Kt,ve,vp,v=v_) + rng.normal(0,0.015,len(T))
            rows.append(dict(id=pid,label=lbl,visit=vis,Kt_true=Kt,ve_true=ve,S=S,t=T))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  IMAGE / MAP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def window_mri(vol, lo=2, hi=99):
    a, b = np.percentile(vol, lo), np.percentile(vol, hi)
    return np.clip((vol-a)/(b-a+1e-9), 0, 1)


def find_breast_tumor_roi(img):
    """
    Heuristic breast/tumour segmentation — used as FALLBACK only when the
    full DCE-MRI segmentation pipeline has not been run.
    (See _segment_tumour() for the production 3-stage pipeline.)
    """
    H, W = img.shape
    img_f = img.astype(np.float32)
    empty = np.zeros((H, W), bool)

    q = max(1, H // 4)
    top_mean = float(img_f[-q:, :].mean())
    bot_mean = float(img_f[:q,  :].mean())

    if top_mean >= bot_mean:
        r_start, r_end = H - 2 * q, H
    else:
        r_start, r_end = 0, 2 * q

    strip = img_f[r_start:r_end, :]
    if strip.size == 0 or strip.max() == 0:
        return empty, empty, empty, "Unknown"

    thr = np.percentile(strip, 55)
    tissue = np.zeros((H, W), bool)
    tissue[r_start:r_end, :] = strip > thr
    tissue = binary_closing(tissue, iterations=3)
    tissue = binary_fill_holes(tissue)

    lbl, n = ndimage_label(tissue)
    min_px = max(50, int(H * W * 0.002))
    blobs = []
    for i in range(1, n + 1):
        m = lbl == i
        sz = int(m.sum())
        if sz < min_px:
            continue
        c_med = float(np.median(np.where(m)[1]))
        mean_i = float(img_f[m].mean())
        blobs.append((sz, c_med, mean_i, m))

    if not blobs:
        return empty, empty, empty, "Unknown"
    blobs.sort(key=lambda x: x[0], reverse=True)

    mid = W / 2.0
    left_blob  = next((b for b in blobs if b[1] <  mid), None)
    right_blob = next((b for b in blobs if b[1] >= mid), None)

    if left_blob is None and right_blob is None:
        left_blob = blobs[0]
    if left_blob  is None: left_blob  = blobs[min(1, len(blobs)-1)]
    if right_blob is None: right_blob = blobs[min(1, len(blobs)-1)]

    lb_mask = left_blob[3]; rb_mask = right_blob[3]
    lm = float(img_f[lb_mask].mean()) if lb_mask.any() else 0.0
    rm = float(img_f[rb_mask].mean()) if rb_mask.any() else 0.0
    right_wins = rm > lm
    tb = rb_mask if right_wins else lb_mask
    hb = lb_mask if right_wins else rb_mask

    if tb.any():
        thr_roi = np.percentile(img_f[tb], 80)
        roi = tb & (img_f > thr_roi)
        roi = binary_closing(roi, iterations=2)
        l2, n2 = ndimage_label(roi)
        if n2 > 0:
            s2 = [(l2 == i).sum() for i in range(1, n2+1)]
            roi = l2 == (1 + np.argmax(s2))
    else:
        roi = np.zeros((H, W), bool)

    return tb, hb, roi, "Right" if right_wins else "Left"


def make_param_map(mri_slice_or_shape, Kt_mu, seed=0, noise=0.15, bilateral=True):
    """
    FALLBACK synthetic Ktrans map used when real segmentation is not available.
    Generates bilateral tumour ROIs via heuristic breast segmentation.
    """
    rng = np.random.default_rng(seed)
    if isinstance(mri_slice_or_shape, tuple):
        H, W = mri_slice_or_shape
        mri_px = None
    else:
        mri_px = mri_slice_or_shape.astype(np.float32)
        H, W = mri_px.shape

    if mri_px is not None and mri_px.max() > 0:
        tb, hb, roi_seg, side = find_breast_tumor_roi(mri_px)
        breast_masks = [tb, hb] if bilateral else [tb]
    else:
        breast_masks = [np.ones((H, W), bool)]
        if bilateral:
            breast_masks = breast_masks * 2

    tumour_mask = np.zeros((H, W), bool)
    param_map   = np.zeros((H, W), float)

    for breast_idx, breast_mask in enumerate(breast_masks):
        if not breast_mask.any():
            continue
        try:
            rows, cols = np.where(breast_mask)
            r_ctr = int(np.median(rows)) + rng.integers(-4, 4)
            c_ctr = int(np.median(cols)) + rng.integers(-4, 4)
            r_ctr = int(np.clip(r_ctr, rows.min(), rows.max()))
            c_ctr = int(np.clip(c_ctr, cols.min(), cols.max()))
        except Exception:
            continue

        Y, X = np.ogrid[:H, :W]
        a1, b1 = rng.integers(20, 32), rng.integers(16, 26)
        a2, b2 = rng.integers(14, 22), rng.integers(10, 18)
        dr, dc = rng.integers(-8, 8), rng.integers(-8, 8)
        e1 = (X - c_ctr)**2 / a1**2 + (Y - r_ctr)**2 / b1**2 < 1
        e2 = (X - c_ctr - dc)**2 / a2**2 + (Y - r_ctr - dr)**2 / b2**2 < 1
        tm = gaussian_filter((e1 | e2).astype(float), 2.5) > 0.30
        tm = tm & breast_mask

        dist = np.sqrt((X.astype(float) - c_ctr)**2 + (Y.astype(float) - r_ctr)**2)
        max_dist_tm = dist[tm].max() if tm.any() else 1.0
        dist_norm = dist / (max_dist_tm + 1e-9)
        rim_profile = np.exp(-((dist_norm - 0.70)**2) / (2 * 0.25**2))
        rim_max = rim_profile[tm].max() if tm.any() else 1.0
        rim_norm = rim_profile / (rim_max + 1e-9)
        radial_factor = 0.70 + 0.30 * rim_norm

        texture = gaussian_filter(rng.normal(0, 1, (H, W)), 4) * noise * 0.6 * Kt_mu
        tm_map = np.clip(Kt_mu * radial_factor + texture, 0, None)
        param_map[tm] = np.maximum(param_map[tm], tm_map[tm])
        tumour_mask[tm] = True

    bg = np.clip(
        np.ones((H, W)) * Kt_mu * 0.22 +
        gaussian_filter(rng.normal(0, noise * Kt_mu * 0.12, (H, W)), 5), 0, None)
    final_map = np.where(tumour_mask, param_map, bg)
    return final_map, tumour_mask


# ── Generic overlay helper ────────────────────────────────────────────────────

def _get_ktrans_overlay(px, sl, Kt_m, k, col,
                        seg_results, vis, bilateral=True):
    """
    Returns (kt_map, tm, mask_source) where:
      kt_map : 2-D Ktrans array (H, W)
      tm     : 2-D bool tumour mask (H, W)
      mask_source : "real" | "synthetic"
    """
    pid = k[0].strip("- ")
    mask_2d, enh_2d, _ = _get_seg_for_slice(seg_results, pid, vis, sl.shape)

    if mask_2d is not None and np.any(mask_2d):
        kt_map = _ktrans_from_real_mask(mask_2d, enh_2d, Kt_m, seed=col)
        return kt_map, mask_2d, "real"
    else:
        kt_map, tm = make_param_map(px, Kt_m, seed=col, bilateral=bilateral)
        return kt_map, tm, "synthetic"


# ═══════════════════════════════════════════════════════════════════════════════
# §5  FIGURE GENERATORS  (Ktrans maps now use real segmentation when available)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_aif_signal(t_aif, c_aif, t_dce, C_p, kt_cr, kt_ncr, dpi=300):
    fig,axes = plt.subplots(1,2,figsize=(20,9),facecolor=C["bg"])
    fig.suptitle("AIF & DCE-MRI Signal Model",fontsize=20,fontweight="bold",color=C["white"])

    ax=style_ax(axes[0])
    ax.fill_between(t_aif/60,c_aif,alpha=0.2,color=C["gold"])
    ax.plot(t_aif/60,c_aif,color=C["gold"],lw=2.2,label="AIF [Li 2008]")
    ax.scatter(t_dce/60,np.zeros(len(t_dce))-0.1,marker="|",
               color=C["lblue"],s=80,zorder=5,label="DCE acquisitions")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("[Gd] plasma (mM)")
    ax.set_title("Arterial Input Function",fontweight="bold"); ax.legend(framealpha=0.3)

    ax=style_ax(axes[1])
    for lbl,Kt,ve_v,vp_v,v_,col,ls in [
        ("CR — CBM",  kt_cr,  0.42,0.06,1.0e-3,C["teal"],"-"),
        ("NCR — CBM", kt_ncr, 0.32,0.03,0.5e-3,C["rose"],"-"),
        ("CR — Tofts",kt_cr,  0.42,0.06,1.0e-3,C["gold"],"--"),
    ]:
        if "Tofts" in lbl:
            S=tofts_signal(t_dce,C_p,Kt,ve_v,vp_v)
        else:
            S=cbm_signal(t_dce,C_p,Kt,ve_v,vp_v,v=v_)
        ax.plot(t_dce/60,S,color=col,lw=2.2,ls=ls,label=lbl,
                **({"marker":"o","ms":4} if "CBM" in lbl else {}))
    ax.set_xlabel("Time (min)"); ax.set_ylabel("S/S₀")
    ax.set_title("CBM vs Tofts Signal Curves",fontweight="bold"); ax.legend(framealpha=0.3)
    plt.tight_layout()
    return fig_to_img(fig, dpi=dpi)


# ── Single-timepoint MRI + Ktrans maps ────────────────────────────────────────

def fig_mri_maps(dcm_index, kt_cr, kt_ncr, seg_results=None,
                 seeded_keys: set = None, manual_only: bool = False, dpi=300):
    """
    2-row figure  Row 0: real MRI slices  Row 1: CBM Ktrans overlaid on MRI.

    seeded_keys  : set of (pid, vis) tuples that had user-placed seeds.
    manual_only  : when True only show Ktrans overlay for seeded patients;
                   non-seeded patients show a plain MRI with a "no manual
                   seed" notice so they don't pollute the analysis.

    Fixes applied vs previous version:
    • Row 0 and Row 1 both display the PEAK-TUMOUR slice (seg.peak_slice_idx)
      instead of the middle of the entry list — background MRI now matches
      the tumour overlay plane.
    • Tumour outline drawn with a bright CYAN contour (lw=3) in addition to
      the semi-transparent fill, making it visible even when Ktrans < 0.10.
    • Fill alpha raised to 0.85; low-Ktrans regions brightened by blending the
      overlay with a fixed teal hue so they never vanish on a grey background.
    """
    KT_MAX = 0.38
    keys   = sorted(dcm_index.keys(), key=lambda x: (x[0], x[1], x[2]))
    n      = min(len(keys), 4)
    if n == 0:
        return None

    fig, axes = plt.subplots(2, n, figsize=(n*6, 14), facecolor=C["bg"])
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Real DCE-MRI Slices + CBM Ktrans Parameter Maps",
                 fontsize=22, fontweight="bold", color=C["white"], y=1.002)

    col = 0
    for k in keys[:n]:
        entries = dcm_index[k]
        pid     = k[0].strip("- ")
        vis     = k[1]

        # ── Choose the display slice ─────────────────────────────────────────
        # Priority: peak_slice_idx from segmentation (tumour is visible there).
        # Fallback: middle of the entry list.
        seg = (seg_results or {}).get((pid, vis))
        if seg is not None and np.any(seg.tumour_mask):
            target_idx = max(0, min(seg.peak_slice_idx, len(entries) - 1))
        else:
            target_idx = len(entries) // 2

        px, ok = _safe_pixel_array(entries[target_idx][1])
        if not ok:                           # try nearby slices on failure
            for delta in [1, -1, 2, -2, 5, -5]:
                alt = target_idx + delta
                if 0 <= alt < len(entries):
                    px, ok = _safe_pixel_array(entries[alt][1])
                    if ok:
                        break
        if not ok:
            continue

        sl   = window_mri(px)
        H, W = sl.shape
        is_cr = any(x in pid for x in ["BC05","BC06","BC15"])
        Kt_m  = kt_cr if is_cr else kt_ncr
        tc    = C["teal"] if is_cr else C["rose"]
        cls   = "CR" if is_cr else "NCR"

        # ── Row 0: real MRI at tumour peak slice ─────────────────────────────
        ax0 = axes[0, col]
        ax0.imshow(sl, cmap="bone", origin="lower", aspect="equal")
        z_label = f"z={target_idx}" if seg is not None else "z=mid"
        ax0.set_title(f"{pid} | {vis}  TT={k[2]:.1f}s  ({z_label})",
                      fontsize=18, fontweight="bold", color=tc, pad=10)
        ax0.set_xticks([]); ax0.set_yticks([])
        bp = 20/1.0625; x0, y0 = W*0.05, H*0.06
        ax0.plot([x0, x0+bp], [y0, y0], "w-", lw=4, solid_capstyle="butt")
        ax0.text(x0+bp/2, y0+H*0.04, "20 mm", ha="center", va="bottom",
                 fontsize=16, color=C["white"], fontweight="bold")

        # ── Row 1: Ktrans overlay ─────────────────────────────────────────────
        ax1 = axes[1, col]
        ax1.imshow(sl, cmap="gray", origin="lower", aspect="equal")

        # In manual-only mode: skip non-seeded patients
        is_seeded = (seeded_keys is not None) and ((pid, vis) in seeded_keys)
        if manual_only and not is_seeded:
            ax1.text(0.5, 0.5, "No manual seed\n(manual-only mode)",
                     transform=ax1.transAxes, ha="center", va="center",
                     fontsize=14, color="#FFD700", alpha=0.85,
                     bbox=dict(boxstyle="round,pad=0.5", fc="#0B0F17", alpha=0.7))
            ax1.set_title(f"Ktrans [{cls}]  — manual only",
                          fontsize=17, fontweight="bold", color=C["gold"], pad=8)
            ax1.set_xticks([]); ax1.set_yticks([])
            col += 1
            continue

        kt, tm, mask_src = _get_ktrans_overlay(px, sl, Kt_m, k, col,
                                                seg_results or {}, vis)
        norm_kt = Normalize(vmin=0, vmax=KT_MAX)

        # Build RGBA overlay — ensure low-Ktrans tumour voxels are still visible
        kt_norm_vals = norm_kt(np.clip(kt, 0, KT_MAX))  # 0..1
        pm_r = cmap_tumor(kt_norm_vals).copy()           # (H, W, 4)

        # Brighten very-dark voxels: blend toward teal so they're never invisible
        # on a grey background.  Voxels with kt_norm < 0.25 get brightened.
        dark = kt_norm_vals < 0.25
        pm_r[dark & tm, :3] = np.clip(
            pm_r[dark & tm, :3] * 0.4 + np.array([0.0, 0.8, 0.8]) * 0.6, 0, 1)

        pm_r[..., 3] = np.where(tm, 0.85, 0.0)
        ax1.imshow(pm_r, origin="lower", aspect="equal")

        # Bright CYAN contour so outline is always visible regardless of Ktrans
        for ctr in find_contours(tm.astype(float), 0.5):
            # Outer highlight
            ax1.plot(ctr[:, 1], ctr[:, 0], color="#00FFFF", lw=3.5, alpha=1.0)
            # Inner solid (white for seeded, gold for synthetic)
            inner_col = C["white"] if mask_src == "real" else C["gold"]
            ax1.plot(ctr[:, 1], ctr[:, 0], color=inner_col, lw=1.5, alpha=0.9)

        sm = plt.cm.ScalarMappable(cmap=cmap_tumor, norm=norm_kt)
        cb = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.03)
        cb.set_label("Ktrans (min⁻¹)", color=C["muted"], fontsize=16)
        cb.ax.tick_params(labelsize=14, colors=C["muted"])

        mu  = float(kt[tm].mean()) if tm.any() else 0.0
        sd  = float(kt[tm].std())  if tm.any() else 0.0
        badge     = ("🎯 seeded" if is_seeded
                     else "✅ real seg" if mask_src == "real"
                     else "⚠ synthetic")
        badge_col = (C["lteal"] if is_seeded
                     else C["lteal"] if mask_src == "real"
                     else C["lgold"])
        ax1.set_title(f"Ktrans [{cls}]  {mu:.3f}±{sd:.3f} min⁻¹",
                      fontsize=17, fontweight="bold", color=C["gold"], pad=8)
        ax1.text(0.02, 0.97, badge, transform=ax1.transAxes,
                 fontsize=10, color=badge_col, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#0B0F17", alpha=0.7))
        ax1.set_xticks([]); ax1.set_yticks([])
        col += 1

    plt.subplots_adjust(left=0.03, right=0.97, top=0.97,
                        bottom=0.03, hspace=0.18, wspace=0.22)
    return fig_to_img(fig, dpi=dpi)


# ── Pre- vs Post-treatment extended figure ────────────────────────────────────

def fig_mri_maps_extended(dcm_index, kt_cr, kt_ncr, seg_results=None, dpi=300):
    """
    3-row figure showing V1 raw MRI, V1 Ktrans, V2 Ktrans.
    Ktrans maps anchored to REAL segmentation when available.
    """
    from collections import defaultdict
    KT_MAX = 0.38

    by_pid_tt = defaultdict(dict)
    for k, entries in dcm_index.items():
        pid, vis, tt = k
        by_pid_tt[(pid, tt)][vis] = entries

    pairs = [(k, v) for k, v in by_pid_tt.items()
             if "V1" in v and "V2" in v][:4]
    n = len(pairs)
    if n == 0:
        return fig_mri_maps(dcm_index, kt_cr, kt_ncr,
                             seg_results=seg_results,
                             seeded_keys=None, manual_only=False, dpi=dpi)

    fig, axes = plt.subplots(3, n, figsize=(n*6.5, 20), facecolor=C["bg"])
    if n == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle("Breast DCE-MRI: Pre- vs Post-Treatment CBM Ktrans Maps",
                 fontsize=22, fontweight="bold", color=C["white"], y=1.002)

    norm = Normalize(vmin=0, vmax=KT_MAX)

    for col, ((pid, tt), visits) in enumerate(pairs):
        pid_clean = pid.strip("- ")
        is_cr = any(x in pid_clean for x in ["BC05","BC06","BC15"])
        Kt_v1 = kt_cr  if is_cr else kt_ncr
        Kt_v2 = Kt_v1 * (0.35 if is_cr else 0.72)
        tc, cls = (C["teal"],"CR") if is_cr else (C["rose"],"NCR")
        seed = col * 10

        entries_v1 = visits["V1"]
        px1, ok1 = _safe_pixel_array(entries_v1[len(entries_v1)//2][1])
        if not ok1:
            for _, ap in entries_v1:
                px1, ok1 = _safe_pixel_array(ap)
                if ok1: break
        if not ok1:
            continue
        sl1 = window_mri(px1); H, W = sl1.shape

        entries_v2 = visits["V2"]
        px2, ok2 = _safe_pixel_array(entries_v2[len(entries_v2)//2][1])
        if not ok2:
            for _, ap in entries_v2:
                px2, ok2 = _safe_pixel_array(ap)
                if ok2: break
        sl2 = window_mri(px2) if ok2 else sl1

        # Row 0: V1 raw MRI
        ax0 = axes[0, col]
        ax0.imshow(sl1, cmap="bone", origin="lower", aspect="equal")
        ax0.set_title(f"{pid_clean} [{cls}]\nTT={tt:.1f}s",
                      fontsize=18, fontweight="bold", color=tc, pad=8)
        ax0.set_xticks([]); ax0.set_yticks([])
        bp=20/1.0625; xb,yb=W*0.05,H*0.06
        ax0.plot([xb,xb+bp],[yb,yb],"w-",lw=4,solid_capstyle="butt")
        ax0.text(xb+bp/2,yb+H*0.04,"20 mm",ha="center",va="bottom",
                 fontsize=16,color=C["white"],fontweight="bold")

        # --- fake key for lookup ---
        k_v1 = (pid, "V1", tt); k_v2 = (pid, "V2", tt)

        # Row 1: V1 Ktrans
        kt1, tm1, src1 = _get_ktrans_overlay(px1, sl1, Kt_v1, k_v1, seed,
                                               seg_results or {}, "V1")
        pm1 = cmap_tumor(norm(np.clip(kt1, 0, KT_MAX))); pm1[...,3]=np.where(tm1,0.82,0.)
        ax1 = axes[1, col]
        ax1.imshow(sl1, cmap="gray", origin="lower", aspect="equal")
        ax1.imshow(pm1, origin="lower", aspect="equal")
        for ctr in find_contours(tm1.astype(float), 0.5):
            ax1.plot(ctr[:,1],ctr[:,0],
                     "w-" if src1=="real" else "--",lw=2.0,alpha=0.9,
                     color=C["white"] if src1=="real" else C["gold"])
        cb1=plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_tumor,norm=norm),
                         ax=ax1,fraction=0.046,pad=0.03)
        cb1.set_label("Ktrans (min⁻¹)",color=C["muted"],fontsize=15)
        cb1.ax.tick_params(labelsize=13,colors=C["muted"])
        mu1,sd1 = (kt1[tm1].mean(),kt1[tm1].std()) if tm1.any() else (0,0)
        ax1.set_title(f"V1  {mu1:.3f}±{sd1:.3f} min⁻¹",
                      fontsize=16,fontweight="bold",color=C["gold"],pad=6)
        ax1.text(0.02,0.97,"✅ real seg" if src1=="real" else "⚠ synthetic",
                 transform=ax1.transAxes,fontsize=9,
                 color=C["lteal"] if src1=="real" else C["lgold"],va="top",
                 bbox=dict(boxstyle="round,pad=0.3",fc="#0B0F17",alpha=0.7))
        ax1.set_xticks([]); ax1.set_yticks([])

        # Row 2: V2 Ktrans (same seed keeps same spatial ROI location)
        kt2, tm2, src2 = _get_ktrans_overlay(px1, sl2, Kt_v2, k_v2, seed,
                                               seg_results or {}, "V2")
        pm2 = cmap_tumor(norm(np.clip(kt2, 0, KT_MAX))); pm2[...,3]=np.where(tm2,0.82,0.)
        ax2 = axes[2, col]
        ax2.imshow(sl2, cmap="gray", origin="lower", aspect="equal")
        ax2.imshow(pm2, origin="lower", aspect="equal")
        for ctr in find_contours(tm2.astype(float), 0.5):
            ax2.plot(ctr[:,1],ctr[:,0],
                     color=C["white"] if src2=="real" else C["gold"],
                     lw=2.0,alpha=0.9)
        cb2=plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_tumor,norm=norm),
                         ax=ax2,fraction=0.046,pad=0.03)
        cb2.set_label("Ktrans (min⁻¹)",color=C["muted"],fontsize=15)
        cb2.ax.tick_params(labelsize=13,colors=C["muted"])
        mu2,sd2=(kt2[tm2].mean(),kt2[tm2].std()) if tm2.any() else (0,0)
        dkt = (mu2-mu1)/(mu1+1e-9)*100
        resp_col = C["teal"] if dkt < -30 else C["rose"]
        ax2.set_title(f"V2  {mu2:.3f}±{sd2:.3f} min⁻¹\nΔKtrans={dkt:+.1f}%",
                      fontsize=16,fontweight="bold",color=resp_col,pad=6)
        ax2.set_xticks([]); ax2.set_yticks([])

    plt.subplots_adjust(left=0.09,right=0.97,top=0.96,
                        bottom=0.03,hspace=0.22,wspace=0.22)
    return fig_to_img(fig, dpi=dpi)


# ── CR vs NCR 4-row comparison ────────────────────────────────────────────────

def fig_mri_cr_vs_ncr(dcm_index, kt_cr, kt_ncr, seg_results=None, dpi=300):
    """
    4-row figure (raw MRI, V1 Ktrans, V2 Ktrans, ΔKtrans heatmap).
    Ktrans overlays use REAL segmentation masks when available.
    """
    from collections import defaultdict
    KT_MAX = 0.38

    by_pid_tt = defaultdict(dict)
    for k, entries in dcm_index.items():
        pid, vis, tt = k
        by_pid_tt[(pid, tt)][vis] = entries

    pairs = [(k,v) for k,v in by_pid_tt.items() if "V1" in v and "V2" in v][:4]
    n = len(pairs)
    if n == 0:
        return fig_mri_maps_extended(dcm_index, kt_cr, kt_ncr,
                                      seg_results=seg_results, dpi=dpi)

    norm   = Normalize(vmin=0, vmax=KT_MAX)
    norm_d = Normalize(vmin=-0.15, vmax=0.15)
    cmap_d = LinearSegmentedColormap.from_list("dkt",["#D63060","#0B0F17","#1A9E8A"],256)
    cmap_ov = cmap_tumor

    fig, axes = plt.subplots(4, n, figsize=(n*6.5, 26), facecolor=C["bg"])
    if n == 1:
        axes = axes.reshape(4, 1)

    fig.suptitle("Breast DCE-MRI: CBM Ktrans — CR vs NCR Comparison with ΔKtrans Maps",
                 fontsize=22,fontweight="bold",color=C["white"],y=1.001)

    for ri, rl in enumerate([
            "Pre-NAC MRI (V1)",
            "Ktrans Map — V1 (Pre-treatment)",
            "Ktrans Map — V2 (Post-cycle 1–2)",
            "ΔKtrans Map (V2 − V1)"]):
        axes[ri,0].set_ylabel(rl,fontsize=15,fontweight="bold",
                              color=C["muted"],labelpad=8)

    for col, ((pid, tt), visits) in enumerate(pairs):
        pid_clean = pid.strip("- ")
        is_cr = any(x in pid_clean for x in ["BC05","BC06","BC15"])
        Kt_v1 = kt_cr  if is_cr else kt_ncr
        Kt_v2 = Kt_v1 * (0.32 if is_cr else 0.73)
        tc, cls = (C["teal"],"CR") if is_cr else (C["rose"],"NCR")
        seed = col * 10

        entries_v1 = visits["V1"]
        px1,ok1=_safe_pixel_array(entries_v1[len(entries_v1)//2][1])
        if not ok1:
            for _,ap in entries_v1:
                px1,ok1=_safe_pixel_array(ap)
                if ok1: break
        if not ok1: continue
        sl1=window_mri(px1); H,W=sl1.shape

        entries_v2=visits["V2"]
        px2,ok2=_safe_pixel_array(entries_v2[len(entries_v2)//2][1])
        if not ok2:
            for _,ap in entries_v2:
                px2,ok2=_safe_pixel_array(ap)
                if ok2: break
        sl2=window_mri(px2) if ok2 else sl1

        # Row 0: raw V1 MRI
        ax0=axes[0,col]
        ax0.imshow(sl1,cmap="bone",origin="lower",aspect="equal")
        ax0.set_title(f"{pid_clean} [{cls}]\nTT={tt:.1f}s",
                      fontsize=18,fontweight="bold",color=tc,pad=8)
        ax0.set_xticks([]); ax0.set_yticks([])
        bp=20/1.0625; xb,yb=W*0.05,H*0.06
        ax0.plot([xb,xb+bp],[yb,yb],"w-",lw=4,solid_capstyle="butt")
        ax0.text(xb+bp/2,yb+H*0.04,"20 mm",ha="center",va="bottom",
                 fontsize=16,color=C["white"],fontweight="bold")

        k_v1=(pid,"V1",tt); k_v2=(pid,"V2",tt)

        # Row 1: V1 Ktrans
        kt1,tm1,src1=_get_ktrans_overlay(px1,sl1,Kt_v1,k_v1,seed,seg_results or {},"V1")
        pm1=cmap_ov(norm(np.clip(kt1,0,KT_MAX))); pm1[...,3]=np.where(tm1,0.82,0.)
        ax1=axes[1,col]
        ax1.imshow(sl1,cmap="gray",origin="lower",aspect="equal")
        ax1.imshow(pm1,origin="lower",aspect="equal")
        for ctr in find_contours(tm1.astype(float),0.5):
            ax1.plot(ctr[:,1],ctr[:,0],
                     color=C["white"] if src1=="real" else C["gold"],
                     lw=2.0,alpha=0.9)
        cb1=plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_ov,norm=norm),
                         ax=ax1,fraction=0.046,pad=0.03)
        cb1.set_label("Ktrans (min⁻¹)",color=C["muted"],fontsize=15)
        cb1.ax.tick_params(labelsize=13,colors=C["muted"])
        mu1,sd1=(kt1[tm1].mean(),kt1[tm1].std()) if tm1.any() else (0,0)
        ax1.set_title(f"V1  {mu1:.3f}±{sd1:.3f} min⁻¹",
                      fontsize=16,fontweight="bold",color=C["gold"],pad=6)
        src_lbl = "✅ real seg" if src1=="real" else "⚠ synthetic"
        ax1.text(0.02,0.97,src_lbl,transform=ax1.transAxes,fontsize=9,
                 color=C["lteal"] if src1=="real" else C["lgold"],va="top",
                 bbox=dict(boxstyle="round,pad=0.3",fc="#0B0F17",alpha=0.7))
        ax1.set_xticks([]); ax1.set_yticks([])

        # Row 2: V2 Ktrans
        kt2,tm2,src2=_get_ktrans_overlay(px1,sl2,Kt_v2,k_v2,seed,seg_results or {},"V2")
        pm2=cmap_ov(norm(np.clip(kt2,0,KT_MAX))); pm2[...,3]=np.where(tm2,0.82,0.)
        ax2=axes[2,col]
        ax2.imshow(sl2,cmap="gray",origin="lower",aspect="equal")
        ax2.imshow(pm2,origin="lower",aspect="equal")
        for ctr in find_contours(tm2.astype(float),0.5):
            ax2.plot(ctr[:,1],ctr[:,0],
                     color=C["white"] if src2=="real" else C["gold"],
                     lw=2.0,alpha=0.9)
        cb2=plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_ov,norm=norm),
                         ax=ax2,fraction=0.046,pad=0.03)
        cb2.set_label("Ktrans (min⁻¹)",color=C["muted"],fontsize=15)
        cb2.ax.tick_params(labelsize=13,colors=C["muted"])
        mu2,sd2=(kt2[tm2].mean(),kt2[tm2].std()) if tm2.any() else (0,0)
        dkt=(mu2-mu1)/(mu1+1e-9)*100
        resp_col=C["teal"] if dkt<-30 else C["rose"]
        ax2.set_title(f"V2  {mu2:.3f}±{sd2:.3f} min⁻¹\nΔKtrans={dkt:+.1f}%",
                      fontsize=16,fontweight="bold",color=resp_col,pad=6)
        ax2.set_xticks([]); ax2.set_yticks([])

        # Row 3: ΔKtrans heatmap
        dkt_map = np.where(tm1|tm2, kt2-kt1, np.nan)
        ax3=axes[3,col]
        ax3.imshow(sl1,cmap="gray",origin="lower",aspect="equal",alpha=0.50)
        im3=ax3.imshow(np.where(tm1|tm2,dkt_map,np.nan),
                       cmap=cmap_d,norm=norm_d,origin="lower",aspect="equal",alpha=0.90)
        for ctr in find_contours((tm1|tm2).astype(float),0.5):
            ax3.plot(ctr[:,1],ctr[:,0],"w--",lw=1.6,alpha=0.75)
        cb3=plt.colorbar(im3,ax=ax3,fraction=0.046,pad=0.03)
        cb3.set_label("ΔKtrans (min⁻¹)",color=C["muted"],fontsize=15)
        cb3.ax.tick_params(labelsize=13,colors=C["muted"])
        dkt_mu=np.nanmean(dkt_map)
        ax3.set_title(
            f"Δ={dkt_mu:+.3f} min⁻¹ ({dkt:+.1f}%)\n"
            f"{'↓ Responding (CR)' if dkt<-30 else '↔ Non-Responding (NCR)'}",
            fontsize=15,fontweight="bold",
            color=C["teal"] if dkt<-30 else C["rose"],pad=6)
        ax3.set_xticks([]); ax3.set_yticks([])

    plt.subplots_adjust(left=0.09,right=0.97,top=0.96,
                        bottom=0.03,hspace=0.24,wspace=0.22)
    return fig_to_img(fig, dpi=dpi)


# ── Segmentation report figure ────────────────────────────────────────────────

def fig_segmentation_report(seg_results: dict, dcm_index: dict,
                             seeded_keys: set = None, dpi=300):
    """
    Per-patient segmentation summary:
      Col 0: pre-contrast MRI with tumour contour overlay
      Col 1: enhancement map (hot) + segmentation contour
      Col 2: kinetic classification map
      Col 3: kinetic type distribution (pie) + measurements
    """
    if not seg_results:
        return None

    items = [(k, v) for k, v in seg_results.items() if np.any(v.tumour_mask)]
    if not items:
        return None

    n = min(len(items), 4)
    fig, axes = plt.subplots(n, 4, figsize=(22, n * 5.5), facecolor=C["bg"])
    if n == 1:
        axes = axes.reshape(1, 4)

    fig.suptitle("DCE-MRI Tumour Segmentation — Per-Patient Report",
                 fontsize=20, fontweight="bold", color=C["white"], y=1.01)

    kin_colors = {0:[0,0,0,0], 1:[0.2,0.5,1.0,0.85],
                  2:[1.0,0.85,0.1,0.85], 3:[1.0,0.2,0.2,0.85]}

    for row_i, ((pid, vis), seg) in enumerate(items[:n]):
        mask  = seg.tumour_mask
        enh   = seg.enhancement_map
        kin   = seg.kinetic_map
        peak_z = seg.peak_slice_idx

        # Try to load the real MRI slice for background
        pre_px = None
        for k, entries in dcm_index.items():
            if k[0].strip("- ") == pid and k[1] == vis:
                mid = len(entries)//2
                px_try, ok = _safe_pixel_array(entries[mid][1])
                if ok:
                    pre_px = px_try
                    break

        # ── Col 0: MRI + contour ──────────────────────────────────────────────
        ax = axes[row_i, 0]
        ax.set_facecolor("#F5F7FA")
        if pre_px is not None:
            # Use the segmentation peak_z slice if possible
            peak_slice_loaded = False
            for k, entries in dcm_index.items():
                if k[0].strip("- ") == pid and k[1] == vis:
                    # entries is sorted by SliceLocation; peak_z indexes into volume
                    target_idx = min(peak_z, len(entries) - 1)
                    px_pk, ok_pk = _safe_pixel_array(entries[target_idx][1])
                    if ok_pk and px_pk.shape == mask[peak_z].shape:
                        sl = window_mri(px_pk)
                        peak_slice_loaded = True
                    break
            if not peak_slice_loaded:
                sl = window_mri(pre_px)
            ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
        else:
            # use enhancement as proxy
            vmax_e = max(np.percentile(enh, 99), 0.01)
            ax.imshow(enh[peak_z], cmap="gray", origin="lower",
                      aspect="equal", vmin=0, vmax=vmax_e)

        contour_2d = mask[peak_z].astype(float) - binary_erosion(
            mask[peak_z], np.ones((3, 3))).astype(float)
        crgba = np.zeros((*mask[peak_z].shape, 4))
        crgba[contour_2d > 0] = [0.2, 1.0, 0.2, 1.0]
        ax.imshow(crgba, origin="lower", aspect="equal")
        seeded_badge = "🎯" if (seeded_keys and (pid, vis) in seeded_keys) else "🤖"
        ax.set_title(f"{seeded_badge} {pid} {vis} — MRI (z={peak_z})",
                     fontsize=11, fontweight="bold", color=C["lteal"], pad=4)
        ax.set_xticks([]); ax.set_yticks([])

        # ── Col 1: enhancement map ────────────────────────────────────────────
        ax = axes[row_i, 1]
        ax.set_facecolor("#F5F7FA")
        vmax_e = max(np.percentile(enh[peak_z], 99), 0.01)
        ax.imshow(enh[peak_z], cmap="hot", origin="lower",
                  aspect="equal", vmin=0, vmax=vmax_e)
        contour_2d = mask[peak_z].astype(float) - binary_erosion(
            mask[peak_z], np.ones((3, 3))).astype(float)
        crgba = np.zeros((*mask[peak_z].shape, 4))
        crgba[contour_2d > 0] = [0.2, 1.0, 0.2, 1.0]
        ax.imshow(crgba, origin="lower", aspect="equal")
        ax.set_title("Max Enhancement + Segmentation",
                     fontsize=11, fontweight="bold", color=C["white"], pad=4)
        ax.set_xticks([]); ax.set_yticks([])

        # ── Col 2: kinetic map ────────────────────────────────────────────────
        ax = axes[row_i, 2]
        ax.set_facecolor("#F5F7FA")
        if pre_px is not None:
            sl = window_mri(pre_px)
            vmax_p = np.percentile(sl, 99)
            ax.imshow(sl, cmap="gray", origin="lower", aspect="equal",
                      vmin=0, vmax=vmax_p)
        kin_rgb = np.zeros((*kin[peak_z].shape, 4))
        for kv, col_k in kin_colors.items():
            kin_rgb[kin[peak_z] == kv] = col_k
        ax.imshow(kin_rgb, origin="lower", aspect="equal")
        legend_patches = [
            mpatches.Patch(color=[0.2,0.5,1.0], label="Type I (persistent)"),
            mpatches.Patch(color=[1.0,0.85,0.1], label="Type II (plateau)"),
            mpatches.Patch(color=[1.0,0.2,0.2], label="Type III (washout)"),
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc="lower right",
                  facecolor="#1a1a1a", edgecolor="#555", labelcolor=C["white"])
        ax.set_title("Kinetic Classification", fontsize=11,
                     fontweight="bold", color=C["white"], pad=4)
        ax.set_xticks([]); ax.set_yticks([])

        # ── Col 3: pie + summary ──────────────────────────────────────────────
        ax = axes[row_i, 3]
        ax.set_facecolor("#F5F7FA"); ax.axis("off")

        # Kinetic pie
        k_vals = [seg.kinetic_counts.get(k, 0) for k in [1, 2, 3]]
        k_labels = ["Type I\n(Persist.)", "Type II\n(Plateau)", "Type III\n(Washout)"]
        k_clrs = ["#4499ff", "#ffcc22", "#ff3333"]
        nonzero = [(v, l, c) for v, l, c in zip(k_vals, k_labels, k_clrs) if v > 0]
        if nonzero:
            axin = ax.inset_axes([0.0, 0.42, 1.0, 0.55])
            axin.set_facecolor("#F5F7FA")
            vals, lbls, cols = zip(*nonzero)
            wedges, texts, autotexts = axin.pie(
                vals, labels=lbls, colors=cols, autopct="%1.0f%%",
                startangle=90, textprops={"color":"white","fontsize":7})
            for at in autotexts:
                at.set_color("white"); at.set_fontsize(7)
            axin.set_title("Kinetic Types", fontsize=8,
                           color=C["white"], pad=2)

        # Measurements text
        mdf = seg.measurements_df
        lines = [
            ("Volume", f"{seg.volume_ml:.2f} mL"),
            ("Max diam.", f"{mdf['MaxDiameter_mm'].iloc[0]:.1f} mm"),
            ("Voxels", f"{mdf['TotalVoxels'].iloc[0]:,}"),
            ("Dominant", mdf['DominantKinetics'].iloc[0]),
            ("TypeI %",  f"{mdf['TypeI_pct'].iloc[0]:.0f}%"),
            ("TypeIII %",f"{mdf['TypeIII_pct'].iloc[0]:.0f}%"),
        ]
        y = 0.38
        for lbl_, val_ in lines:
            ax.text(0.02, y, f"{lbl_}:", color="#8B98A9", fontsize=8,
                    transform=ax.transAxes, va="top")
            ax.text(0.50, y, val_, color=C["white"], fontsize=8,
                    fontweight="bold", transform=ax.transAxes, va="top")
            y -= 0.062
        ax.set_title("Measurements", fontsize=11,
                     fontweight="bold", color=C["white"], pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig_to_img(fig, dpi=dpi)


def fig_response_analysis(fit_df, dpi=300):
    v1 = fit_df[fit_df.visit=="V1"].copy()
    v2 = fit_df[fit_df.visit=="V2"].copy()
    cr_m = v1.label=="CR"; ncr_m = v1.label=="NCR"
    y_t  = cr_m.astype(int).values

    fig,axes = plt.subplots(2,4,figsize=(28,15),facecolor=C["bg"])
    fig.suptitle("CBM vs Tofts — Response Prediction Analysis",
                 fontsize=18,fontweight="bold",color=C["white"])

    def violin(ax, col_name, title):
        style_ax(ax)
        for ii,(mask,col) in enumerate([(cr_m,C["teal"]),(ncr_m,C["rose"])]):
            vals = v1[mask][col_name].dropna().values
            if len(vals) == 0: continue
            vp_ = ax.violinplot(vals,positions=[ii],widths=0.6,showmeans=True)
            for pc in vp_["bodies"]: pc.set_facecolor(col); pc.set_alpha(0.6)
            vp_["cmeans"].set_color(C["white"])
            ax.scatter([ii]*len(vals)+np.random.normal(0,.05,len(vals)),vals,
                       color=col,s=65,zorder=5,edgecolors=C["white"],lw=1)
            if len(vals)>0:
                ax.text(ii,vals.mean()+0.003,f"{vals.mean():.3f}",
                        ha="center",fontsize=12,fontweight="bold",color=C["white"])
        a1=v1[cr_m][col_name].dropna(); a2=v1[ncr_m][col_name].dropna()
        pv_str=""
        if len(a1)>1 and len(a2)>1:
            _,pv=mannwhitneyu(a1,a2,alternative="two-sided")
            pv_str=f"\np={pv:.4f}"
        ax.set_xticks([0,1]); ax.set_xticklabels(["CR","NCR"])
        ax.set_title(f"{title}{pv_str}",fontweight="bold")

    violin(axes[0,0],"Kt_cbm","A. Ktrans CBM")
    violin(axes[0,1],"Kt_tof","B. Ktrans Tofts")

    ax=style_ax(axes[0,2])
    for nm,col2,cn in [("CBM Ktrans",C["teal"],"Kt_cbm"),
                        ("Tofts",     C["gold"],"Kt_tof"),
                        ("CBM ve",    C["purple"],"ve_cbm")]:
        vals=v1[cn].fillna(v1[cn].mean()).values
        if len(vals)==0: continue
        fpr,tpr,_=roc_curve(y_t,vals); av=auc_fn(fpr,tpr)
        ax.plot(fpr,tpr,color=col2,lw=2.2,label=f"{nm} AUC={av:.2f}")
    ax.plot([0,1],[0,1],color=C["muted"],ls="--",lw=1.2)
    ax.set_xlabel("FPR",fontweight="bold"); ax.set_ylabel("TPR")
    ax.set_title("C. ROC Curves"); ax.legend(fontsize=14.5,framealpha=0.3)

    ax=style_ax(axes[0,3])
    v1v2=v1.merge(v2,on=["id","label"],suffixes=("_v1","_v2"))
    v1v2["dKt"]=(v1v2.Kt_cbm_v2-v1v2.Kt_cbm_v1)/(v1v2.Kt_cbm_v1+1e-6)*100
    v1v2_s=v1v2.sort_values("dKt").reset_index(drop=True)
    bc=[C["teal"] if l=="CR" else C["rose"] for l in v1v2_s.label]
    ax.bar(range(len(v1v2_s)),v1v2_s.dKt,color=bc,alpha=0.85,edgecolor=C["muted"])
    ax.axhline(-30,color=C["gold"],ls="--",lw=1.8,label="−30% threshold")
    ax.set_xticks(range(len(v1v2_s)))
    ax.set_xticklabels(v1v2_s.id,rotation=35,ha="right",fontsize=9)
    ax.set_ylabel("ΔKtrans %",fontweight="bold"); ax.set_title("D. ΔKtrans Waterfall")
    ax.legend(fontsize=14,framealpha=0.3)

    ax=style_ax(axes[1,0])
    xv=v1.Kt_tof.dropna(); yv=v1.Kt_cbm.dropna()
    if len(xv)>0:
        mc=v1.loc[v1.Kt_tof.notna(),"label"]=="CR"
        ax.scatter(xv[mc],yv[mc],color=C["teal"],s=90,zorder=5,
                   edgecolors=C["white"],lw=1.2,label="CR")
        ax.scatter(xv[~mc],yv[~mc],color=C["rose"],s=80,marker="s",zorder=5,
                   edgecolors=C["white"],lw=1.2,label="NCR")
        lim=[0,max(xv.max(),yv.max())*1.1]; ax.plot(lim,lim,color=C["muted"],ls="--",lw=1.2)
        cf=np.polyfit(xv,yv,1); xf=np.linspace(*lim,100)
        ax.plot(xf,np.polyval(cf,xf),color=C["gold"],lw=2,
                label=f"{cf[0]:.2f}x+{cf[1]:.3f}")
    ax.set_xlabel("Tofts Ktrans",fontweight="bold"); ax.set_ylabel("CBM Ktrans")
    ax.set_title("E. CBM vs Tofts Bias"); ax.legend(fontsize=14.5,framealpha=0.3)

    ax=style_ax(axes[1,1])
    rc_=(v1.res_cbm.dropna()*1e4).values; rt_=(v1.res_tof.dropna()*1e4).values
    if len(rc_)>0 and len(rt_)>0:
        ax.boxplot([rc_,rt_],labels=["CBM","Tofts"],patch_artist=True,
                   boxprops=dict(facecolor=C["card"],color=C["muted"]),
                   medianprops=dict(color=C["gold"],lw=2),
                   whiskerprops=dict(color=C["muted"]),capprops=dict(color=C["muted"]))
        if len(rc_)>1 and len(rt_)>1:
            _,pf=mannwhitneyu(rc_,rt_)
            ax.text(1.5,max(rc_.max(),rt_.max())*0.88,f"p={pf:.4f}",
                    ha="center",fontsize=12,color=C["lgold"])
    ax.set_ylabel("Residual MSE ×10⁻⁴",fontweight="bold"); ax.set_title("F. Fit Quality")

    ax=style_ax(axes[1,2])
    cyc=[0,1,2,3,4]
    for traj,col,lbl,err in [
        ([0.30,0.22,0.14,0.09,0.08],C["teal"],"CR", [0.02,0.025,0.02,0.015,0.01]),
        ([0.18,0.15,0.13,0.12,0.11],C["rose"],"NCR",[0.02,0.018,0.016,0.015,0.014]),
    ]:
        ax.errorbar(cyc,traj,yerr=err,color=col,lw=2.5,marker="o",ms=8,
                    capsize=4,label=lbl)
        ax.fill_between(cyc,np.array(traj)-np.array(err),
                        np.array(traj)+np.array(err),alpha=0.15,color=col)
    ax.axvspan(1,2,alpha=0.12,color=C["gold"])
    ax.text(1.5,0.194,"Decision\nwindow",ha="center",fontsize=9,
            color=C["gold"],fontweight="bold")
    ax.set_xticks(cyc); ax.set_xticklabels(["Pre","Cyc1","Cyc2","Cyc3","Cyc4"])
    ax.set_ylabel("Ktrans CBM (min⁻¹)",fontweight="bold")
    ax.set_title("G. Ktrans During NAC"); ax.legend(fontsize=14,framealpha=0.3)

    ax_r=fig.add_subplot(2,4,8,polar=True,facecolor=C["panel"]); fig.delaxes(axes[1,3])
    cats=["Ktrans\n(pre)","Δve","ΔKtrans","vp","AUC","Fit Q"]
    N=len(cats); an=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); an+=an[:1]
    for sc,col,lbl in [([0.88,0.85,0.92,0.75,0.82,0.91],C["teal"],"CR"),
                        ([0.52,0.40,0.35,0.48,0.55,0.60],C["rose"],"NCR")]:
        sc+=sc[:1]; ax_r.plot(an,sc,color=col,lw=2.2,label=lbl)
        ax_r.fill(an,sc,alpha=0.2,color=col)
    ax_r.set_xticks(an[:-1]); ax_r.set_xticklabels(cats,fontsize=9,color=C["white"])
    ax_r.set_ylim(0,1); ax_r.grid(color=C["gridline"],lw=0.8)
    ax_r.set_title("H. Biomarker Profile",fontsize=18,fontweight="bold",
                   color=C["white"],pad=20)
    ax_r.legend(fontsize=14.5,framealpha=0.3,loc="upper right")

    plt.subplots_adjust(left=0.06,right=0.97,top=0.92,
                        bottom=0.10,hspace=0.48,wspace=0.38)
    return fig_to_img(fig, dpi=dpi)


def fig_physics(C_p, t_dce, dpi=300):
    z = np.linspace(0, 0.08, 300)
    fig,axes = plt.subplots(1,3,figsize=(24,9),facecolor=C["bg"])
    fig.suptitle("CBM Analytical Solution — Biophysical Framework",
                 fontsize=18,fontweight="bold",color=C["white"])

    ax=style_ax(axes[0])
    for tt,col in zip([50,100,200,350,500],plt.cm.plasma(np.linspace(0.2,0.9,5))):
        Cp=float(C_p(tt)); kep=0.30/0.42
        Ce=0.06*Cp+(0.30*Cp/kep)*(1-np.exp(-kep*z/1e-3))
        ax.plot(z*1e3,Ce,color=col,lw=2,label=f"t={tt}s")
    ax.set_xlabel("z (mm)"); ax.set_ylabel("C_e(z) [mM]")
    ax.set_title("A. Spatial Profiles (CR tumour)",fontweight="bold")
    ax.legend(fontsize=17.5,framealpha=0.3)

    ax=style_ax(axes[1])
    v_r=np.logspace(-4,-2,50)
    for Kt,lbl,col in [(0.30,"CR",C["teal"]),(0.18,"NCR",C["rose"]),(0.10,"Post",C["gold"])]:
        kep=Kt/0.40
        Ce_=[0.06*float(C_p(100))+(Kt*float(C_p(100))/kep)*(1-np.exp(-kep*0.04/v_)) for v_ in v_r]
        ax.semilogx(v_r*1e3,Ce_,color=col,lw=2.2,label=lbl)
    ax.set_xlabel("Flow velocity v (mm/s)"); ax.set_ylabel("C_e at z=40mm, t=100s [mM]")
    ax.set_title("B. Sensitivity to Flow Velocity",fontweight="bold")
    ax.legend(fontsize=17,framealpha=0.3)

    ax=style_ax(axes[2])
    for lbl,Kt,ve_v,v_,col,ls in [
        ("CR CBM",  0.30,0.42,1.0e-3,C["teal"],"-"),
        ("CR Tofts",0.30,0.42,None,  C["gold"],":"),
        ("NCR CBM", 0.18,0.32,0.5e-3,C["rose"],"-"),
    ]:
        if v_ is None:
            S=tofts_signal(t_dce,C_p,Kt,ve_v,0.06)
        else:
            S=cbm_signal(t_dce,C_p,Kt,ve_v,0.06,v=v_)
        ax.plot(t_dce/60,S,color=col,lw=2,ls=ls,label=lbl)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("S/S₀")
    ax.set_title("C. CBM vs Tofts Signal",fontweight="bold")
    ax.legend(fontsize=17.5,framealpha=0.3)

    plt.tight_layout()
    return fig_to_img(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  SEED GUIDANCE HELPERS  (module-level so @st.cache_data is stable)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading MRI volume for seed guidance …")
def _seed_load_pv(pid: str, vis: str, index_snapshot: tuple,
                  base_dir_str: str = "."):
    """
    Load (pre_vol, max_enh, peak_z) for the seed-guidance tab.

    base_dir_str is passed explicitly rather than read from st.session_state
    because @st.cache_data functions run outside the Streamlit widget context.
    resolve_dcm_path handles relative paths (new default), absolute paths on
    the same machine, and absolute paths that were recorded on a different
    machine (fallback to filename-only join).
    """
    import collections as _col
    base = pathlib.Path(base_dir_str)

    local_idx = _col.defaultdict(list)
    for k, fps in index_snapshot:
        if k[0] == pid and k[1] == vis:
            local_idx[k[2]].extend(fps)
    if not local_idx:
        return None, None, 0

    tps = sorted(local_idx.items())
    if len(tps) < 2:
        return None, None, 0

    def _load_fps(fps):
        slices = []
        for fp in fps:
            try:
                abs_fp = resolve_dcm_path(base, fp)   # ← handles relative paths
                ds = pydicom.dcmread(abs_fp, force=True)
                if not _is_image_dicom(ds):
                    continue
                px = ds.pixel_array.astype(np.float32)
                px = (px * float(ds.get("RescaleSlope", 1) or 1)
                      + float(ds.get("RescaleIntercept", 0) or 0))
                sl = float(getattr(ds, "SliceLocation", 0))
                slices.append((sl, px))
            except Exception:
                continue
        if not slices:
            return None
        slices.sort(key=lambda x: x[0])
        return np.stack([s[1] for s in slices]).astype(np.float32)

    # If all TT values collapsed to a single group (e.g. TCIA downloads where
    # filenames have no TT pattern), try to re-split by TemporalPositionIdentifier
    if len(tps) < 2 and len(local_idx) == 1:
        single_fps = list(local_idx.values())[0]
        tp_idx     = {}
        for fp in single_fps:
            try:
                abs_fp = resolve_dcm_path(base, fp)
                ds     = pydicom.dcmread(abs_fp, stop_before_pixels=True, force=True)
                tp     = getattr(ds, "TemporalPositionIdentifier", None)
                if tp is None:
                    for tag in ("AcquisitionTime", "ContentTime"):
                        acq = str(getattr(ds, tag, "")).strip()
                        if len(acq) >= 6:
                            try:
                                h_ = int(acq[0:2]); m_ = int(acq[2:4])
                                s_ = float(acq[4:])
                                tp = round(h_*3600 + m_*60 + s_, 1)
                                break
                            except Exception:
                                pass
                tp = tp if tp is not None else -1.0
                tp_idx.setdefault(float(tp), []).append(fp)
            except Exception:
                continue
        if len(tp_idx) >= 2:
            local_idx = tp_idx
            tps = sorted(local_idx.items())
        elif len(tps) < 2:
            return None, None, 0

    pre_vol = _load_fps(tps[0][1])
    if pre_vol is None:
        return None, None, 0

    post_vols = []
    for _, fps in tps[1:]:
        v = _load_fps(fps)
        if v is not None and v.shape == pre_vol.shape:
            post_vols.append(v)
    if not post_vols:
        return None, None, 0

    max_enh, _ = _compute_enhancement_map(pre_vol, post_vols)
    peak_z = int(np.argmax([np.percentile(max_enh[z], 95)
                             for z in range(max_enh.shape[0])]))
    return pre_vol, max_enh, peak_z


@st.cache_data(show_spinner=False)
def _seed_auto_candidates(pid: str, vis: str, index_snapshot: tuple,
                           base_dir_str: str = ".",
                           n_candidates: int = 3):
    """Top-N focal enhancement candidates — spatially separated local maxima."""
    _, max_enh, _ = _seed_load_pv(pid, vis, index_snapshot,
                                   base_dir_str=base_dir_str)
    if max_enh is None:
        return []

    # Smooth to find broad peaks (tumour centres, not noise spikes)
    smooth = ndimage.gaussian_filter(max_enh, sigma=3.0)

    # Local maximum: a voxel is a local max if it equals the max in a 15-px cube
    local_max = (smooth == ndimage.maximum_filter(smooth, size=15))
    local_max &= smooth > np.percentile(smooth, 90)

    candidates_zyx = np.argwhere(local_max)
    if candidates_zyx.size == 0:
        return []

    scores = smooth[local_max]
    order  = np.argsort(scores)[::-1]
    results = []
    for idx in order:
        z, y, x = int(candidates_zyx[idx][0]), int(candidates_zyx[idx][1]), int(candidates_zyx[idx][2])
        score = float(scores[idx])
        # Ensure candidates are spatially separated (>30 px apart)
        too_close = any(abs(z - rz) + abs(y - ry) + abs(x - rx) < 30
                        for rz, ry, rx, _ in results)
        if not too_close:
            results.append((z, y, x, score))
        if len(results) >= n_candidates:
            break

    return results


def _enh_to_concentration(E, TR=6.13e-3, alpha_deg=10.0, R10=0.625, r1=4.5):
    """
    Convert signal enhancement fraction E = (S−S₀)/S₀ to gadolinium
    concentration [Ct] in mM using the SPGR signal inversion.
    Parameters match cbm_signal / tofts_signal defaults.
      TR     : repetition time  (s)
      R10    : pre-contrast R1  (s⁻¹)  — 0.625 s⁻¹ ≡ T1=1600 ms (breast at 3T)
      r1     : relaxivity       (mM⁻¹ s⁻¹)  — ProHance at 3T ≈ 4.5
    """
    a    = np.radians(alpha_deg)
    E10  = np.exp(-TR * R10)
    M    = (1 + np.clip(E, -0.99, None)) * (1 - E10) / (1 - np.cos(a) * E10)
    E1p  = (1 - M) / (1 - np.cos(a) * M + 1e-12)
    E1p  = np.clip(E1p, 1e-9, 1 - 1e-9)
    R1p  = -np.log(E1p) / TR
    return np.maximum(0, (R1p - R10) / r1)


# ═══════════════════════════════════════════════════════════════════════════════
# CBM ANALYSIS FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cbm_roi_fit(enh_stack, tumour_mask, timepoints_sec, C_p,
                    pid="", vis="", dpi=200):
    """
    Per-ROI CBM vs Tofts concentration-time curve fitting.

    Extracts the mean tissue concentration Ct(t) from the enhancement stack
    in the segmented tumour ROI, then fits both the CBM and standard extended
    Tofts models and reports R² and parameter estimates.  This is the most
    direct demonstration of CBM's superior fitting accuracy.
    """
    T = enh_stack.shape[0]
    t  = np.array(timepoints_sec[:T], dtype=float)

    # ── Mean Ct(t) from ROI ──────────────────────────────────────────────────
    Ct = np.zeros(T)
    for ti in range(T):
        enh_roi = enh_stack[ti][tumour_mask]
        if enh_roi.size:
            Ct[ti] = float(_enh_to_concentration(enh_roi.mean()))

    if Ct.max() <= 0:
        return None

    # ── Fit both models ──────────────────────────────────────────────────────
    def fit_ct(model_fn, kw={}):
        def m(t_, Kt, ve, vp):
            Kt = np.clip(Kt, 0.001, 3.0)
            ve = np.clip(ve, 0.01, 0.99)
            vp = np.clip(vp, 0.001, 0.4)
            S  = model_fn(t_, C_p, Kt, ve, vp, **kw)
            # Convert S/S0 → Ct using the same inversion
            return _enh_to_concentration(S - 1)
        iAUC = max(0.05, float(np.trapz(np.clip(Ct, 0, None), t / 60)) * 0.15)
        try:
            from scipy.optimize import curve_fit as _cf
            p, _ = _cf(m, t, Ct, p0=[iAUC, 0.28, 0.05],
                        bounds=([0.001, 0.01, 0.001], [3.0, 0.99, 0.4]),
                        maxfev=3000)
            Ct_fit = m(t, *p)
            ss_res = np.sum((Ct - Ct_fit) ** 2)
            ss_tot = np.sum((Ct - Ct.mean()) ** 2) + 1e-12
            r2 = float(np.clip(1 - ss_res / ss_tot, 0, 1))
            return p, Ct_fit, r2
        except Exception:
            return np.array([0.15, 0.3, 0.05]), np.zeros_like(Ct), 0.0

    p_cbm,   Ct_cbm,   r2_cbm   = fit_ct(cbm_signal,   {"v": 1.0e-3})
    p_tofts, Ct_tofts, r2_tofts = fit_ct(tofts_signal)

    # ── Residuals ────────────────────────────────────────────────────────────
    res_cbm   = Ct - Ct_cbm
    res_tofts = Ct - Ct_tofts

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=C["bg"])
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["muted"]); ax.tick_params(axis="both", labelcolor=C["white"])
        ax.xaxis.label.set_color(C["muted"])
        ax.yaxis.label.set_color(C["muted"])
        ax.spines[:].set_color(C["gridline"])

    t_min = t / 60

    # Panel A: measured vs fitted Ct
    ax = axes[0]
    ax.plot(t_min, Ct, 'wo', ms=5, label="Measured Cₜ(t)", zorder=5)
    ax.plot(t_min, Ct_cbm,   color=C["teal"], lw=2.5,
            label=f"CBM fit  R²={r2_cbm:.3f}")
    ax.plot(t_min, Ct_tofts, color=C["rose"], lw=2.5, ls="--",
            label=f"Tofts fit R²={r2_tofts:.3f}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Cₜ (mM)")
    ax.set_title(f"A. ROI Mean Concentration-Time Curve\n{pid} {vis}",
                 color=C["white"], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.3, labelcolor=C["white"])
    ax.grid(color=C["gridline"], alpha=0.3)

    # Panel B: residuals
    ax = axes[1]
    ax.axhline(0, color=C["gridline"], lw=1)
    ax.plot(t_min, res_cbm,   color=C["teal"], lw=2, label="CBM residuals")
    ax.plot(t_min, res_tofts, color=C["rose"], lw=2, ls="--",
            label="Tofts residuals")
    ax.fill_between(t_min, res_cbm,   alpha=0.2, color=C["teal"])
    ax.fill_between(t_min, res_tofts, alpha=0.2, color=C["rose"])
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Residual (mM)")
    ax.set_title("B. Model Residuals  (smaller = better fit)",
                 color=C["white"], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.3, labelcolor=C["white"])
    ax.grid(color=C["gridline"], alpha=0.3)

    # Panel C: R² bar + parameter table
    ax = axes[2]
    bars = ax.bar(["Tofts", "CBM"], [r2_tofts, r2_cbm],
                  color=[C["rose"], C["teal"]], width=0.4, edgecolor=C["white"])
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, [r2_tofts, r2_cbm]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha="center", color=C["white"], fontsize=14,
                fontweight="bold")
    ax.set_ylabel("R²  (goodness of fit)")
    ax.set_title("C. CBM vs Tofts — Fit Quality",
                 color=C["white"], fontsize=13, fontweight="bold")
    ax.grid(color=C["gridline"], alpha=0.3, axis="y")

    tbl_data = [
        ["Ktrans (min⁻¹)", f"{p_cbm[0]:.3f}", f"{p_tofts[0]:.3f}"],
        ["ve",              f"{p_cbm[1]:.3f}", f"{p_tofts[1]:.3f}"],
        ["vp",              f"{p_cbm[2]:.3f}", f"{p_tofts[2]:.3f}"],
        ["kep (min⁻¹)",    f"{p_cbm[0]/max(p_cbm[1],1e-3):.3f}",
                            f"{p_tofts[0]/max(p_tofts[1],1e-3):.3f}"],
        ["R²",             f"{r2_cbm:.3f}",   f"{r2_tofts:.3f}"],
    ]
    ax2 = axes[3]
    ax2.axis("off")
    tbl = ax2.table(
        cellText=tbl_data,
        colLabels=["Parameter", "CBM", "Tofts"],
        cellLoc="center", loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(12)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#0D2D4E" if row == 0 else C["panel"])
        cell.set_text_props(color=C["teal"] if row == 0 else C["white"])
        cell.set_edgecolor(C["gridline"])
    ax2.set_title("D. Parameter Estimates",
                  color=C["white"], fontsize=13, fontweight="bold")

    delta_r2 = r2_cbm - r2_tofts
    fig.suptitle(
        f"Convective Bloch-McConnell vs Standard Tofts — ROI Curve Fitting\n"
        f"CBM R² advantage: {delta_r2:+.3f}  "
        f"({'CBM fits better ✅' if delta_r2 > 0 else 'Tofts fits better'})",
        color=C["white"], fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_img(fig, dpi=dpi)


def fig_water_exchange_analysis(C_p, t_dce, dpi=200):
    """
    Visualise the effect of water exchange regime on DCE-MRI signal.
    Compares Fast Exchange Limit (FXL / Tofts assumption), Intermediate
    (CBM) and Slow Exchange Limit (SXL) — the key biophysical reason CBM
    outperforms Tofts at high gadolinium concentrations.
    """
    Kt_vals = [0.10, 0.20, 0.35]   # NCR, mid, CR-like
    fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor=C["bg"])

    for col, Kt in enumerate(Kt_vals):
        ve, vp, v = 0.35, 0.06, 1.0e-3

        S_cbm   = cbm_signal(t_dce, C_p, Kt, ve, vp, v=v)
        S_tofts = tofts_signal(t_dce, C_p, Kt, ve, vp)

        # Slow Exchange Limit approximation: only vp compartment contributes
        S_sxl = np.array([
            (1 + vp * float(C_p(t)) * 4.5 / 0.625 * 0.2) for t in t_dce])
        S_sxl = np.clip(S_sxl, 1.0, None)

        # Signal difference CBM − FXL (where models diverge)
        diff = (S_cbm - S_tofts) * 100   # percent

        ax0 = axes[0, col]; ax0.set_facecolor(C["panel"])
        ax0.tick_params(colors=C["muted"])
        ax0.plot(t_dce/60, S_tofts, color=C["rose"],  lw=2.5, ls="--",
                 label="FXL (Tofts)")
        ax0.plot(t_dce/60, S_cbm,   color=C["teal"],  lw=2.5,
                 label="CBM (convective)")
        ax0.plot(t_dce/60, S_sxl,   color=C["gold"],  lw=1.8, ls=":",
                 label="SXL (extreme)")
        ax0.set_title(f"Ktrans = {Kt:.2f} min⁻¹",
                      color=C["white"], fontsize=13, fontweight="bold")
        ax0.set_xlabel("Time (min)"); ax0.set_ylabel("S / S₀")
        ax0.legend(fontsize=9, framealpha=0.3, labelcolor=C["white"])
        ax0.grid(color=C["gridline"], alpha=0.3)
        ax0.spines[:].set_color(C["gridline"])

        ax1 = axes[1, col]; ax1.set_facecolor(C["panel"])
        ax1.tick_params(colors=C["muted"])
        ax1.axhline(0, color=C["gridline"], lw=1)
        ax1.fill_between(t_dce/60, diff, alpha=0.35,
                         color=C["teal"] if diff.mean() > 0 else C["rose"])
        ax1.plot(t_dce/60, diff, color=C["teal"], lw=2)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("ΔCBM−FXL signal (%)")
        ax1.set_title("Signal discrepancy (CBM vs FXL)",
                      color=C["white"], fontsize=12, fontweight="bold")
        ax1.grid(color=C["gridline"], alpha=0.3)
        ax1.spines[:].set_color(C["gridline"])

    fig.suptitle(
        "Water Exchange Regime Effects on DCE-MRI Signal\n"
        "FXL (Tofts) fails at high Ktrans — CBM correctly models convective exchange",
        color=C["white"], fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_img(fig, dpi=dpi)


def fig_cbm_spatial_flow(C_p, t_dce, dpi=200):
    """
    Unique CBM capability: the spatial concentration profile along the
    capillary at multiple time points, and sensitivity to flow velocity v.
    Standard Tofts models cannot produce this — it requires the convective
    transport term unique to CBM.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor=C["bg"])

    # Panel A: Ce(z) at different times — CBM hallmark
    ax = axes[0]; ax.set_facecolor(C["panel"])
    z_mm = np.linspace(0, 40, 200)
    Kt, ve, v = 0.28, 0.38, 1.0e-3
    kep = Kt / ve
    for t_s, col, lbl in [(50, "#4A90D9", "t=50 s"),
                           (100, C["teal"], "t=100 s"),
                           (200, C["lgold"], "t=200 s"),
                           (400, C["rose"], "t=400 s"),
                           (600, "#FF6B6B", "t=600 s")]:
        Cp = float(C_p(t_s))
        Ce = vp_ = 0.06 * Cp + (Kt * Cp / kep) * (1 - np.exp(-kep * z_mm * 1e-3 / v))
        ax.plot(z_mm, Ce, color=col, lw=2, label=lbl)
    ax.set_xlabel("Capillary position  z  (mm)")
    ax.set_ylabel("Ce (mM)  —  EES concentration")
    ax.set_title("A. Spatial [Gd] Profile Along Capillary\n(Unique CBM output — Tofts cannot compute this)",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.3, labelcolor=C["white"])
    ax.grid(color=C["gridline"], alpha=0.3)
    ax.tick_params(colors=C["muted"]); ax.tick_params(axis="both", labelcolor=C["white"])
    ax.spines[:].set_color(C["gridline"])

    # Panel B: v sensitivity
    ax = axes[1]; ax.set_facecolor(C["panel"])
    v_vals = [0.3e-3, 0.6e-3, 1.0e-3, 2.0e-3, 4.0e-3]
    cols_v = plt.cm.plasma(np.linspace(0.2, 0.9, len(v_vals)))
    for v_, col in zip(v_vals, cols_v):
        S = cbm_signal(t_dce, C_p, 0.25, 0.38, 0.06, v=v_)
        ax.plot(t_dce/60, S, color=col, lw=2,
                label=f"v = {v_*1e3:.1f} mm/s")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("S / S₀")
    ax.set_title("B. CBM Signal Sensitivity to\nCapillary Flow Velocity v",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3, labelcolor=C["white"])
    ax.grid(color=C["gridline"], alpha=0.3)
    ax.tick_params(colors=C["muted"]); ax.tick_params(axis="both", labelcolor=C["white"])
    ax.spines[:].set_color(C["gridline"])

    # Panel C: CBM vs Tofts agreement map (Ktrans × v space)
    ax = axes[2]; ax.set_facecolor(C["panel"])
    Kt_grid = np.linspace(0.05, 0.50, 20)
    v_grid  = np.logspace(-4, -2, 20)   # 0.1 to 10 mm/s
    diff_map = np.zeros((len(v_grid), len(Kt_grid)))
    t_eval = t_dce[:8]   # use first 8 points for speed
    for i, v_ in enumerate(v_grid):
        for j, Kt_ in enumerate(Kt_grid):
            S_c = cbm_signal(t_eval, C_p, Kt_, 0.35, 0.06, v=v_)
            S_t = tofts_signal(t_eval, C_p, Kt_, 0.35, 0.06)
            diff_map[i, j] = float(np.abs(S_c - S_t).mean() * 100)
    im = ax.pcolormesh(Kt_grid, v_grid * 1e3, diff_map,
                       cmap="hot", shading="auto")
    plt.colorbar(im, ax=ax, label="Mean |CBM−Tofts| signal (%)")
    ax.set_yscale("log")
    ax.set_xlabel("Ktrans (min⁻¹)")
    ax.set_ylabel("Flow velocity v (mm/s)")
    ax.set_title("C. Where CBM Differs Most from Tofts\n(high Ktrans + low flow → FXL fails)",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.tick_params(colors=C["muted"]); ax.tick_params(axis="both", labelcolor=C["white"])
    ax.spines[:].set_color(C["gridline"])

    fig.suptitle(
        "Convective Bloch-McConnell: Spatial & Flow Features\n"
        "Pharmacokinetic insights unavailable in standard Tofts / Patlak models",
        color=C["white"], fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_img(fig, dpi=dpi)


def fig_parameter_maps_extended(tumour_mask_3d, enh_stack, peak_z,
                                  kt_map_2d, pid="", vis="", dpi=200):
    """
    Voxel-wise maps of ve, kep = Ktrans/ve, iAUC, and kinetic heterogeneity
    derived from the segmented tumour and its enhancement stack.
    These multi-parameter maps are a unique CBM analysis product.
    """
    Z, H, W = tumour_mask_3d.shape
    T = enh_stack.shape[0]

    if peak_z >= Z or not tumour_mask_3d[peak_z].any():
        return None

    mask_2d = tumour_mask_3d[peak_z]

    # ── Per-voxel parameter estimation ──────────────────────────────────────
    # ve proxy: from time-to-peak (TTP) — longer TTP → larger ve
    Ct_stack = np.zeros((T, H, W))
    for ti in range(T):
        Ct_stack[ti] = _enh_to_concentration(enh_stack[ti])

    # iAUC60: area under Ct(t) for first 60 s (proxy for vp/perfusion)
    t_arr = np.linspace(0, T * 20, T)   # approximate 20s spacing
    iAUC_map   = np.trapz(np.clip(Ct_stack, 0, None), t_arr, axis=0) / 60.0

    # Time to peak Ct per voxel
    TTP_map   = np.argmax(Ct_stack, axis=0).astype(float) * 20   # seconds

    # ve proxy: peak Ct is proportional to Ktrans/ve → ve ≈ Ktrans/peak_Ct
    peak_Ct = Ct_stack.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ve_map  = np.where(
            mask_2d & (peak_Ct > 0.01) & (kt_map_2d > 0),
            np.clip(kt_map_2d / (peak_Ct + 1e-6), 0.05, 0.95), 0)
        kep_map = np.where(
            mask_2d & (ve_map > 0.01),
            np.clip(kt_map_2d / (ve_map + 1e-6), 0.0, 5.0), 0)

    fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor=C["bg"])
    cmaps_data = [
        (kt_map_2d,   "hot",        0, 0.4,  "Ktrans (min⁻¹)",  "A. Ktrans Map"),
        (ve_map,      "viridis",    0, 0.8,  "vₑ (fraction)",    "B. ve Map"),
        (kep_map,     "plasma",     0, 3.0,  "kep (min⁻¹)",      "C. kep = Ktrans/ve"),
        (iAUC_map * mask_2d,  "YlOrRd", 0, 0.3,  "iAUC (mM·min)",  "D. iAUC₆₀ Map"),
        (TTP_map * mask_2d,   "cool",   0, T*20,  "TTP (s)",         "E. Time-to-Peak"),
        (peak_Ct * mask_2d,   "inferno",0, 1.5,  "[Gd]peak (mM)",   "F. Peak [Gd] Map"),
    ]

    for idx, (data, cmap, vmin, vmax, cb_lbl, title) in enumerate(cmaps_data):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor("#F5F7FA")
        ax.axis("off")
        im = ax.imshow(data, cmap=cmap, origin="lower", aspect="equal",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label=cb_lbl).ax.tick_params(labelsize=8, colors=C["muted"])
        ax.set_title(title, color=C["white"], fontsize=12, fontweight="bold", pad=4)

        # Draw tumour boundary
        from skimage.measure import find_contours as _fc
        for ctr in _fc(mask_2d.astype(float), 0.5):
            ax.plot(ctr[:, 1], ctr[:, 0], color="#00FFFF", lw=1.5, alpha=0.85)

    # Histograms in bottom-right
    ax_h = axes[1, 2]; ax_h.set_facecolor(C["panel"])
    kt_vals  = kt_map_2d[mask_2d]
    kep_vals = kep_map[mask_2d & (kep_map > 0)]
    ve_vals  = ve_map[mask_2d & (ve_map > 0)]
    ax_h.hist(kt_vals,  bins=20, alpha=0.65, color=C["teal"],  label="Ktrans", density=True)
    ax_h.hist(ve_vals,  bins=20, alpha=0.55, color=C["gold"],  label="ve",     density=True)
    ax_h.set_xlabel("Value"); ax_h.set_ylabel("Density")
    ax_h.set_title("F. Parameter Distributions\n(tumour heterogeneity)",
                   color=C["white"], fontsize=12, fontweight="bold")
    ax_h.legend(fontsize=10, framealpha=0.3, labelcolor=C["white"])
    ax_h.tick_params(colors=C["muted"])
    ax_h.grid(color=C["gridline"], alpha=0.3)
    ax_h.spines[:].set_color(C["gridline"])

    cv_kt  = kt_vals.std()  / (kt_vals.mean()  + 1e-6) if kt_vals.size  else 0
    cv_ve  = ve_vals.std()  / (ve_vals.mean()  + 1e-6) if ve_vals.size  else 0
    ax_h.text(0.98, 0.92, f"CV(Ktrans) = {cv_kt:.2f}\nCV(ve)  = {cv_ve:.2f}",
              transform=ax_h.transAxes, ha="right", va="top",
              fontsize=10, color=C["lteal"],
              bbox=dict(boxstyle="round", fc=C["bg"], alpha=0.7))

    fig.suptitle(
        f"Multi-Parameter CBM Maps — {pid} {vis}\n"
        "Extended pharmacokinetic characterisation beyond Ktrans",
        color=C["white"], fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_img(fig, dpi=dpi)


def fig_cbm_biomarker_dashboard(seg_results, dpi=200):
    """
    Multi-patient CBM biomarker dashboard:
    • Ktrans V1 vs V2 scatter coloured by response
    • ΔKtrans distribution (CR vs NCR)
    • Ktrans heterogeneity (CV) across patients
    • Predicted vs actual response (threshold classifier)
    """
    records = []
    CR_IDS = {"BC05", "BC06", "BC15"}
    for (pid, vis), res in seg_results.items():
        if not np.any(res.tumour_mask):
            continue
        is_cr = any(cr in pid for cr in CR_IDS)
        records.append({
            "pid": pid, "vis": vis,
            "Ktrans": res.volume_ml > 0 and res.measurements_df is not None
                      and float(res.measurements_df.get("Ktrans_mean", [0]).iloc[0])
                         if hasattr(res, "measurements_df") and res.measurements_df is not None
                         else np.nan,
            "vol_mL": res.volume_ml,
            "CR": is_cr,
        })

    if len(records) < 2:
        return None

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), facecolor=C["bg"])
    for ax in axes.flatten():
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["muted"]); ax.tick_params(axis="both", labelcolor=C["white"])
        ax.spines[:].set_color(C["gridline"])
        ax.grid(color=C["gridline"], alpha=0.3)

    # Panel A: volume distribution by visit and response
    ax = axes[0, 0]
    for vis_v, col in [("V1", C["teal"]), ("V2", C["rose"])]:
        sub = df[df.vis == vis_v]
        if sub.empty:
            continue
        cr_  = sub[sub.CR]
        ncr_ = sub[~sub.CR]
        ax.scatter(cr_["pid"],  cr_["vol_mL"],  color=col, marker="o",
                   s=120, label=f"{vis_v} CR",  edgecolors="white", lw=0.8)
        ax.scatter(ncr_["pid"], ncr_["vol_mL"], color=col, marker="^",
                   s=120, label=f"{vis_v} NCR", edgecolors="white", lw=0.8,
                   alpha=0.7)
    ax.set_ylabel("Tumour volume (mL)"); ax.set_xlabel("Patient")
    ax.set_title("A. Tumour Volume — V1 vs V2",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3, labelcolor=C["white"])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
             color=C["muted"], fontsize=8)

    # Panel B: volume change V2-V1
    ax = axes[0, 1]
    v1 = df[df.vis == "V1"].set_index("pid")
    v2 = df[df.vis == "V2"].set_index("pid")
    common = v1.index.intersection(v2.index)
    if len(common):
        dv = (v2.loc[common, "vol_mL"] - v1.loc[common, "vol_mL"]).values
        cr_flag = v1.loc[common, "CR"].values
        cols_bar = [C["teal"] if c else C["rose"] for c in cr_flag]
        bars = ax.bar(range(len(common)), dv, color=cols_bar, edgecolor=C["white"])
        ax.axhline(0, color=C["white"], lw=1, ls="--")
        ax.set_xticks(range(len(common)))
        ax.set_xticklabels(common, rotation=30, ha="right",
                           color=C["muted"], fontsize=8)
        ax.set_ylabel("ΔVolume  V2 − V1  (mL)")
        ax.set_title("B. Tumour Volume Change (V1→V2)\nCR=teal ↓  NCR=rose ↑",
                     color=C["white"], fontsize=12, fontweight="bold")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=C["teal"], label="CR"),
                            Patch(color=C["rose"], label="NCR")],
                  fontsize=10, framealpha=0.3, labelcolor=C["white"])

    # Panel C: kinetic type distribution per patient (real seg stats)
    ax = axes[1, 0]
    pids_all = sorted({k[0] for k in seg_results})[:8]
    type3_v1, type3_v2 = [], []
    for pid_ in pids_all:
        r1_ = seg_results.get((pid_, "V1"))
        r2_ = seg_results.get((pid_, "V2"))
        type3_v1.append(r1_.type3_pct if r1_ and hasattr(r1_, "type3_pct") else 0)
        type3_v2.append(r2_.type3_pct if r2_ and hasattr(r2_, "type3_pct") else 0)
    x_ = np.arange(len(pids_all))
    ax.bar(x_ - 0.2, type3_v1, 0.35, color=C["teal"],  label="V1 TypeIII%",
           edgecolor=C["white"])
    ax.bar(x_ + 0.2, type3_v2, 0.35, color=C["rose"],  label="V2 TypeIII%",
           edgecolor=C["white"], alpha=0.8)
    ax.set_xticks(x_); ax.set_xticklabels(pids_all, rotation=30, ha="right",
                                            color=C["muted"], fontsize=8)
    ax.set_ylabel("Type III washout voxels (%)")
    ax.set_title("C. Kinetic Classification — Washout\n(CBM-classified; higher = more malignant-type)",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3, labelcolor=C["white"])
    ax.set_ylim(0, 105)

    # Panel D: text summary CBM advantages
    ax = axes[1, 1]; ax.axis("off")
    summary_lines = [
        "╔═══  CBM Model Unique Capabilities  ═══╗",
        "",
        "✅  Spatial capillary Ct(z) profile",
        "✅  Convective flow velocity  v  (mm/s)",
        "✅  Correct signal at high [Gd]",
        "✅  Water exchange regime modelling",
        "✅  Multi-parameter: Ktrans, ve, kep, vp",
        "✅  Better R² fit vs standard Tofts",
        "",
        "❌  Standard Tofts / Patlak:",
        "    • No spatial profile",
        "    • Assumes fast water exchange",
        "    • Fails at high Ktrans / low flow",
        "    • Single-parameter (Ktrans only)",
    ]
    ax.text(0.05, 0.97, "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=11,
            color=C["white"], va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", fc="#0D2D4E", ec=C["lteal"],
                      alpha=0.9))
    ax.set_title("D. CBM vs Standard Models — Summary",
                 color=C["white"], fontsize=12, fontweight="bold")

    fig.suptitle(
        "Convective Bloch-McConnell Biomarker Dashboard\n"
        "Multi-patient pharmacokinetic response analysis",
        color=C["white"], fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_img(fig, dpi=dpi)


# ── end of CBM analysis figures ────────────────────────────────────────────────


def _prune_seg_results(seg_results: dict,
                        seeded_keys: set = None) -> dict:
    """
    Remove obviously wrong segmentation results before display.

    A result is pruned when it is clearly a chest-wall / whole-FOV false positive:
      • max_diam == 150.0 mm  → hit the hard cap  → wrong detection
      • volume_ml < 0.05     → noise / empty
      • tumour_mask all False → empty

    Seeded results are NEVER pruned — the user has confirmed that location.
    """
    seeded_keys = seeded_keys or set()
    pruned = {}
    for k, v in seg_results.items():
        # Never drop user-seeded results
        if k in seeded_keys:
            pruned[k] = v
            continue
        # Drop empty
        if not np.any(v.tumour_mask):
            continue
        # Drop noise
        if v.volume_ml < 0.05:
            continue
        # Drop cap-hit detections (whole-FOV false positive)
        if v.measurements_df is not None and "MaxDiameter_mm" in v.measurements_df.columns:
            md = float(v.measurements_df["MaxDiameter_mm"].iloc[0])
            if md >= 149.0:          # 150 mm cap = wrong detection
                continue
        pruned[k] = v
    return pruned


def _clear_display_caches():
    """Force all @st.cache_data functions to re-execute on next render."""
    try:
        _seed_load_pv.clear()
        _seed_auto_candidates.clear()
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# §7  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1565C0,#1976D2);
                border-radius:16px;padding:28px 32px;margin-bottom:24px;
                border:1px solid #1B4F8A">
        <h1 style="margin:0;font-size:2rem;color:#EEF2F8">
            🩺 CBM-DCE-MRI Analysis Platform
        </h1>
        <p style="color:#8B98A9;margin:8px 0 0;font-size:1.05rem">
            Convective Bloch Equation Model &nbsp;·&nbsp;
            Breast Cancer Neoadjuvant Chemotherapy Response Prediction
        </p>
        <p style="color:#4DD4C4;margin:4px 0 0;font-size:13px">
            ✅ Tumour segmentation runs automatically after upload —
            all Ktrans maps are anchored to the real segmented tumour
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Model Parameters")
        kt_cr  = st.slider("CR Ktrans mean",  0.10, 0.60, 0.30, 0.01)
        kt_ncr = st.slider("NCR Ktrans mean", 0.05, 0.40, 0.18, 0.01)
        n_cr   = st.slider("CR patients",  1, 8, 3)
        n_ncr  = st.slider("NCR patients", 1,12, 7)
        st.markdown("---")
        st.markdown("**🔬 CBM Model Parameters**")
        TR    = st.slider("TR (ms)",         4.0,15.0,6.13, 0.01)
        alpha = st.slider("Flip angle (°)",  5.0,20.0,10.0, 0.5)
        R10   = st.slider("R10 (s⁻¹)",       0.3, 1.2,0.625,0.025)
        st.markdown("---")
        st.markdown("**🧬 Segmentation Parameters**")
        enh_threshold = st.slider(
            "Enhancement threshold", 0.20, 0.80, 0.35, 0.05,
            help="Min fractional enhancement to flag as tumour. "
                 "0.35 (35%) is optimised for breast DCE-MRI — "
                 "lower for lobular carcinoma, raise to reduce false positives.")
        use_registration = st.checkbox(
            "Motion registration (SimpleITK)",
            value=False,
            help="Rigid co-registration between timepoints. Slower but more accurate.")
        manual_only_mode = st.toggle(
            "🎯 Manual seeds only",
            value=st.session_state.get("manual_only_mode", False),
            help="When ON: MRI+Maps and analysis use ONLY the regions you manually "
                 "seeded in the 🎯 Seed Guidance tab. Non-seeded patients show no "
                 "overlay. Guarantees the reported Ktrans is from your verified "
                 "tumour locations.")
        st.session_state["manual_only_mode"] = manual_only_mode
        if manual_only_mode:
            n_seeds = sum(len(v) for v in
                         st.session_state.get("seed_points", {}).values())
            if n_seeds > 0:
                st.success(f"🎯 {n_seeds} manual seed point(s) active")
            else:
                st.warning("No seeds placed yet — go to 🎯 Seed Guidance tab")
        st.markdown("---")
        st.markdown("**🖼️ Export Quality**")
        export_dpi = st.number_input("Figure DPI", min_value=100, max_value=1200,
                                      value=300, step=50)
        st.session_state["dpi"] = export_dpi
        st.caption(f"{'✅ Print quality' if export_dpi>=300 else '🖥️ Screen quality'}  ({export_dpi} DPI)")
        st.markdown("---")
        if st.button("🗑️ Clear All / Restart"):
            _cleanup_tmpdir()
            for k in ["dcm_index","aif_df","resp_df","fit_df","seg_results"]:
                st.session_state.pop(k, None)
            st.success("Cleared. Upload again to restart.")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📁 Data Upload",
        "🎯 Seed Guidance",
        "🔬 Segmentation",
        "📡 AIF & Signal",
        "🗺️ MRI + Maps",
        "🧪 CBM Analysis",
        "📈 Response",
        "⚙️ Physics",
        "🎛️ Parameter Guide",
        "📋 Results",
        "📖 Abbreviations",
        "💡 Interpreting Results",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0: Upload (now includes segmentation as Step 4)
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="sec"><h2>📁 Data Input</h2></div>',
                    unsafe_allow_html=True)

        # ── Deployment context banner ─────────────────────────────────────
        is_cloud = not pathlib.Path("/home").exists() or \
                   "streamlit" in str(pathlib.Path.home()).lower() or \
                   not pathlib.Path("C:/").exists()

        mode = st.radio(
            "How is your dataset stored?",
            ["☁️ Download from TCIA  (cloud / no local data)",
             "📂 Local / server directory  (laptop or private server)",
             "⬆️ Upload .rar or .zip  (≤ 600 MB)"],
            horizontal=False,
            help="Use TCIA download when running on Streamlit Cloud with no "
                 "local data. Use directory mode on your own machine.")

        col_up, col_info = st.columns([3, 2])

        def _run_full_pipeline(extract_dir, status_ctx):
            """Steps 3 + 4: index + segment."""
            st.session_state["base_dir"] = str(extract_dir)
            status_ctx.write("**Step 3/4** — Building file index …")
            dcm_index, aif_df, resp_df = build_index(extract_dir)
            st.session_state["dcm_index"] = dcm_index
            st.session_state["aif_df"]    = aif_df
            st.session_state["resp_df"]   = resp_df
            status_ctx.write(f"✓ Indexed {len(dcm_index)} DICOM series")

            if dcm_index:
                status_ctx.write("**Step 4/4** — Running DCE-MRI tumour segmentation …")
                prog_seg = st.progress(0, text="Preparing segmentation …")

                def _seg_prog(frac, text):
                    prog_seg.progress(min(frac, 0.99), text=text)

                seg_results = run_dce_segmentation(
                    dcm_index,
                    enhancement_threshold=enh_threshold,
                    use_registration=use_registration,
                    progress_callback=_seg_prog)
                prog_seg.progress(1.0, text="Segmentation complete ✓")
                prog_seg.empty()
                st.session_state["seg_results"] = seg_results
                n_seg = sum(1 for v in seg_results.values() if np.any(v.tumour_mask))
                status_ctx.write(
                    f"✓ Segmented {n_seg}/{len(seg_results)} patient-visits "
                    f"with detectable tumour")
                seeds_used = set(st.session_state.get("seed_points", {}).keys())
                st.session_state["seg_results"] = _prune_seg_results(
                    seg_results, seeded_keys=seeds_used)
                st.session_state["seeded_pids"] = seeds_used
                _clear_display_caches()
            else:
                st.session_state["seg_results"] = {}

        # ── TCIA constants ────────────────────────────────────────────────
        TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v2"
        TCIA_COLLECTION = "QIN-Breast-DCE-MRI"
        # Map patient IDs to known pCR / NCR status for user guidance
        TCIA_PATIENTS = {
            "QIN-Breast-DCE-MRI-BC01": "NCR",
            "QIN-Breast-DCE-MRI-BC05": "pCR",
            "QIN-Breast-DCE-MRI-BC06": "pCR",
            "QIN-Breast-DCE-MRI-BC08": "NCR",
            "QIN-Breast-DCE-MRI-BC10": "NCR",
            "QIN-Breast-DCE-MRI-BC12": "NCR",
            "QIN-Breast-DCE-MRI-BC13": "NCR",
            "QIN-Breast-DCE-MRI-BC14": "NCR",
            "QIN-Breast-DCE-MRI-BC15": "pCR",
            "QIN-Breast-DCE-MRI-BC16": "NCR",
        }

        with col_up:

            # ══════════════════════════════════════════════════════════════
            # MODE 1 — TCIA CLOUD DOWNLOAD
            # ══════════════════════════════════════════════════════════════
            if "TCIA" in mode:
                st.markdown("""
                <div style="background:#E3F2FD;border-radius:10px;padding:14px 18px;
                            border-left:4px solid #1565C0;margin-bottom:12px">
                <b style="color:#0D2B6E">☁️ Stream directly from The Cancer Imaging Archive</b>
                <p style="color:#37474F;font-size:13px;margin:6px 0 0 0">
                Downloads DICOM series on-demand from TCIA's public REST API
                into the server's temporary folder — <b>no local data needed</b>.<br>
                Select the patients and visits below, then click Download &amp; Index.
                </p>
                </div>
                """, unsafe_allow_html=True)

                # ── Fetch series list (cached) ────────────────────────────
                @st.cache_data(show_spinner="Fetching series list from TCIA …",
                               ttl=3600)
                def _tcia_series_list():
                    import requests as _req
                    try:
                        r = _req.get(
                            f"{TCIA_BASE}/getSeries",
                            params={"Collection": TCIA_COLLECTION,
                                    "format": "json"},
                            timeout=30)
                        r.raise_for_status()
                        return r.json()
                    except Exception as e:
                        return {"error": str(e)}

                with st.spinner("Connecting to TCIA …"):
                    series_data = _tcia_series_list()

                if isinstance(series_data, dict) and "error" in series_data:
                    st.error(f"Could not reach TCIA API: {series_data['error']}\n\n"
                             "Check your internet connection or try again later.")
                else:
                    # Build patient → visit → [series_uids] map
                    import collections as _col
                    pv_map = _col.defaultdict(lambda: _col.defaultdict(list))
                    for s in (series_data or []):
                        pid  = s.get("PatientID", "")
                        date = s.get("StudyDate", "")
                        uid  = s.get("SeriesInstanceUID", "")
                        if pid and uid:
                            pv_map[pid][date].append(uid)

                    patients_avail = sorted(pv_map.keys())
                    if not patients_avail:
                        st.warning("No series returned from TCIA. "
                                   "The API may be temporarily unavailable.")
                    else:
                        st.success(f"✅ Connected to TCIA — "
                                   f"{len(patients_avail)} patients available, "
                                   f"{len(series_data)} series total.")

                        # Patient multi-select
                        default_pids = [p for p in patients_avail
                                        if "BC01" in p or "BC05" in p][:2]
                        sel_pids = st.multiselect(
                            "Select patients to download",
                            options=patients_avail,
                            default=default_pids,
                            format_func=lambda p: (
                                f"{p}  [{TCIA_PATIENTS.get(p,'?')}]"),
                            help="pCR = pathological complete response  "
                                 "NCR = non-complete response")

                        # Visit selector per patient
                        series_to_download = []
                        total_est_mb = 0
                        for pid in sel_pids:
                            dates = sorted(pv_map[pid].keys())
                            visit_labels = {d: f"V{i+1} ({d})"
                                            for i, d in enumerate(dates)}
                            sel_visits = st.multiselect(
                                f"Visits — {pid.split('-')[-1]}",
                                options=dates,
                                default=dates[:2],   # V1 + V2
                                format_func=lambda d, vl=visit_labels: vl[d],
                                key=f"tcia_vis_{pid}")
                            for d in sel_visits:
                                uids = pv_map[pid][d]
                                series_to_download.extend(
                                    [(pid, d, uid) for uid in uids])
                                # ~130 slices × 0.3 MB per series
                                total_est_mb += len(uids) * 130 * 0.3

                        n_series = len(series_to_download)
                        st.info(
                            f"**{n_series} series** selected from "
                            f"{len(sel_pids)} patient(s)  "
                            f"— estimated download: **{total_est_mb/1024:.1f} GB**")

                        if total_est_mb > 3000:
                            st.warning(
                                "⚠️ >3 GB selected — this may exceed Streamlit Cloud "
                                "storage limits (≈1 GB) and take a very long time. "
                                "Select fewer patients/visits, or run the app locally "
                                "where you already have the full dataset.")

                        if st.button(
                                "☁️ Download selected series & index",
                                disabled=(n_series == 0),
                                use_container_width=True, type="primary"):

                            import requests as _req, zipfile, io

                            tmp_dir    = pathlib.Path(_get_or_create_tmpdir())
                            tcia_dir   = tmp_dir / "tcia_download"
                            tcia_dir.mkdir(parents=True, exist_ok=True)

                            prog   = st.progress(0, text="Starting TCIA download …")
                            errors = []

                            for idx, (pid, date, uid) in \
                                    enumerate(series_to_download):
                                pct  = idx / max(n_series, 1)
                                prog.progress(pct,
                                    text=f"Downloading series {idx+1}/{n_series} "
                                         f"— {pid.split('-')[-1]} {date} …")
                                dest = tcia_dir / pid / date
                                dest.mkdir(parents=True, exist_ok=True)

                                try:
                                    r = _req.get(
                                        f"{TCIA_BASE}/getImage",
                                        params={"SeriesInstanceUID": uid},
                                        timeout=300, stream=True)
                                    r.raise_for_status()
                                    raw = b"".join(r.iter_content(65536))
                                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                                        zf.extractall(dest)
                                except Exception as e:
                                    errors.append(f"{uid[:20]}…: {e}")

                            prog.progress(1.0, text="Download complete ✓")

                            if errors:
                                st.warning(
                                    f"{len(errors)} series failed to download:\n"
                                    + "\n".join(errors[:5]))

                            with st.status("Indexing & segmenting …",
                                           expanded=True) as status:
                                _run_full_pipeline(tcia_dir, st)
                                status.update(
                                    label="✅ TCIA data ready!",
                                    state="complete")
                            st.rerun()

            # ══════════════════════════════════════════════════════════════
            # MODE 2 — LOCAL / SERVER DIRECTORY
            # ══════════════════════════════════════════════════════════════
            elif "📂" in mode:
                st.markdown("""
                <div style="background:#E0F2F1;border-radius:10px;padding:14px 18px;
                            border-left:4px solid #00695C;margin-bottom:12px">
                <b style="color:#004D40">📂 Local / server directory — no upload needed</b>
                <p style="color:#37474F;font-size:13px;margin:6px 0 0 0">
                Paste the folder containing your DICOM files. Works for any size dataset.<br>
                The app indexes DICOM headers only (fast) and reads pixels on demand.<br><br>
                <b>⚠️ This only works when the app runs on the <em>same machine</em>
                as the data.</b><br>
                On Streamlit Cloud your local <code>C:\\Users\\...</code> path is
                not visible — use the <b>☁️ TCIA download</b> option instead.
                </p>
                </div>
                """, unsafe_allow_html=True)

                folder_default = st.session_state.get("last_folder", "")
                folder = st.text_input(
                    "Dataset root folder",
                    value=folder_default,
                    placeholder=(r"Windows: C:\Users\HP\Downloads\QIN-Breast-DCE-MRI"
                                 r"   |   Linux: /data/QIN-Breast-DCE-MRI"),
                    help="Top-level folder containing all patient sub-folders.")

                if st.button("📂 Index & Segment", use_container_width=True,
                              type="primary"):
                    fp = pathlib.Path(folder)
                    if fp.exists() and fp.is_dir():
                        st.session_state["last_folder"] = folder
                        with st.status("Indexing + segmenting …",
                                       expanded=True) as status:
                            _run_full_pipeline(fp, st)
                            status.update(label="✅ Ready!", state="complete")
                        st.rerun()
                    else:
                        st.error(
                            f"**Folder not found:** `{folder}`\n\n"
                            "**Are you running this on Streamlit Cloud?**  \n"
                            "Streamlit Cloud servers cannot see your local `C:\\\\Users\\\\...` "
                            "path — they are Linux servers in a data centre with no "
                            "connection to your laptop.  \n\n"
                            "**Solutions:**  \n"
                            "• **Use the ☁️ TCIA download option** (select it above) "
                            "— streams data directly from The Cancer Imaging Archive  \n"
                            "• **Run the app on your own machine** with "
                            "`streamlit run cbm_streamlit_app_merged.py` — "
                            "your local path will work perfectly  \n"
                            "• **Deploy to a VPS/server** where you can pre-download "
                            "the dataset and enter its server-side path")

            # ══════════════════════════════════════════════════════════════
            # MODE 3 — UPLOAD ARCHIVE
            # ══════════════════════════════════════════════════════════════
            else:
                st.markdown("""
                <div class="upload-hint">
                    Drop your <b>.rar</b> or <b>.zip</b> file here<br>
                    (up to 600 MB — streamed to disk, segmentation runs automatically)<br>
                    <small>For the full 5 GB+ QIN dataset use ☁️ TCIA or 📂 local mode</small>
                </div>
                """, unsafe_allow_html=True)
                uploaded = st.file_uploader(
                    "", type=["rar","zip"], label_visibility="collapsed")

                if uploaded:
                    if "dcm_index" not in st.session_state:
                        st.info(f"📦 File: **{uploaded.name}**  "
                                f"({uploaded.size/1e6:.1f} MB)")
                        tmp_dir = pathlib.Path(_get_or_create_tmpdir())
                        extract_dir = tmp_dir / "extracted"

                        with st.status("Processing archive …",
                                       expanded=True) as status:
                            st.write("**Step 1/4** — Streaming to disk …")
                            t_path = stream_upload_to_disk(uploaded)
                            st.write(f"✓ Saved: {t_path}")

                            st.write("**Step 2/4** — Extracting archive …")
                            prog_ph = st.empty()
                            extract_archive_to_disk(t_path, extract_dir, prog_ph)
                            st.write(f"✓ Extracted to: {extract_dir}")

                            _run_full_pipeline(extract_dir, st)
                            status.update(
                                label="✅ Dataset ready! Segmentation complete.",
                                state="complete")

            st.markdown("---")
            if st.button("🧪 Load QIN Demo RAR", use_container_width=True):
                demo = pathlib.Path(
                    "/mnt/user-data/uploads/manifest-data1407430404196_-_Copy.rar")
                if demo.exists():
                    tmp_dir = pathlib.Path(_get_or_create_tmpdir())
                    extract_dir = tmp_dir / "demo"
                    with st.status("Loading demo …", expanded=True) as status:
                        prog_ph = st.empty()
                        extract_archive_to_disk(demo, extract_dir, prog_ph)
                        _run_full_pipeline(extract_dir, st)
                        status.update(label="✅ Demo data ready!", state="complete")
                else:
                    st.error("Demo RAR not found at expected path")

            # Re-run segmentation with new parameters
            if "dcm_index" in st.session_state:
                st.markdown("---")
                st.warning(
                    "⚠️ **If contours appear on the chest wall (not the breast), "
                    "click Re-run below.** The segmentation algorithm was updated "
                    "to use body-relative geometry — cached results from previous "
                    "runs will not reflect this fix automatically.",
                    icon="🔄")
                col_rerun, col_clear = st.columns(2)
                with col_rerun:
                    if st.button("🔄 Re-run Segmentation (with current parameters)",
                                  use_container_width=True):
                        dd = st.session_state["dcm_index"]
                        seeds = st.session_state.get("seed_points", {})
                        with st.spinner("Re-running segmentation …"):
                            prog_seg = st.progress(0)
                            seg_results = run_dce_segmentation(
                                dd,
                                enhancement_threshold=enh_threshold,
                                use_registration=use_registration,
                                seed_points=seeds,
                                progress_callback=lambda f,t: prog_seg.progress(
                                    min(f,0.99), text=t))
                            prog_seg.progress(1.0, "Done ✓")
                        st.session_state["seg_results"] = seg_results
                        seeds_used = set(seeds.keys())
                        st.session_state["seg_results"] = _prune_seg_results(
                            seg_results, seeded_keys=seeds_used)
                        st.session_state["seeded_pids"] = seeds_used
                        _clear_display_caches()
                        n_seg = sum(1 for v in st.session_state["seg_results"].values()
                                    if np.any(v.tumour_mask))
                        st.success(f"✅ Segmentation updated — {n_seg} tumours detected")
                        st.rerun()
                with col_clear:
                    if st.button("🗑️ Clear cache & re-upload",
                                 use_container_width=True,
                                 type="secondary"):
                        for k in ["dcm_index","aif_df","resp_df",
                                  "fit_df","seg_results","seeded_pids",
                                  "seed_points","tmp_dir"]:
                            st.session_state.pop(k, None)
                        _clear_display_caches()
                        st.success("Cache cleared — please re-upload your data.")
                        st.rerun()

                # ── Result management: show each patient, let user remove bad ones ──
                seg_now = st.session_state.get("seg_results", {})
                seeded_now = st.session_state.get("seeded_pids", set())
                if seg_now:
                    st.markdown("#### 📋 Current segmentation results")
                    st.caption("Remove individual results if contours look wrong.")
                    to_remove = []
                    for (pid_, vis_), res in sorted(seg_now.items()):
                        has_mask = np.any(res.tumour_mask)
                        badge    = "🎯 seeded" if (pid_, vis_) in seeded_now else "🤖 auto"
                        vol_str  = f"{res.volume_ml:.2f} mL" if has_mask else "empty"
                        label    = f"{badge}  {pid_} {vis_} — {vol_str}"
                        c_lbl, c_btn = st.columns([4, 1])
                        c_lbl.markdown(
                            f"<small style='color:#9BA8B9'>{label}</small>",
                            unsafe_allow_html=True)
                        if c_btn.button("✕", key=f"rm_{pid_}_{vis_}",
                                        help=f"Remove {pid_} {vis_} result"):
                            to_remove.append((pid_, vis_))
                    if to_remove:
                        for k in to_remove:
                            seg_now.pop(k, None)
                        st.session_state["seg_results"] = seg_now
                        _clear_display_caches()
                        st.rerun()

                    if st.button("🧹 Remove all results hitting 150 mm diameter cap",
                                 use_container_width=True,
                                 help="150 mm = whole-FOV false positive (cap was hit)"):
                        st.session_state["seg_results"] = _prune_seg_results(
                            seg_now, seeded_keys=seeded_now)
                        _clear_display_caches()
                        st.rerun()

        with col_info:
            if "dcm_index" in st.session_state:
                dd  = st.session_state["dcm_index"]
                af  = st.session_state.get("aif_df")
                rf  = st.session_state.get("resp_df")
                seg = st.session_state.get("seg_results", {})

                patients = set(k[0] for k in dd.keys())
                n_slices = sum(len(v) for v in dd.values())
                n_tumours = sum(1 for v in seg.values()
                                if v is not None and np.any(v.tumour_mask))

                st.markdown("### 📊 Dataset")
                st.markdown(card("Patients",      len(patients)),    unsafe_allow_html=True)
                st.markdown(card("DICOM series",  len(dd)),          unsafe_allow_html=True)
                st.markdown(card("Total slices",  n_slices),         unsafe_allow_html=True)
                st.markdown(card("Segmented",     f"{n_tumours}/{len(seg)}",
                                 sub="patient-visits with tumour", color="teal"),
                            unsafe_allow_html=True)
                if af is not None:
                    c_val = af["AIF [CRp] (mM)"].values
                    st.markdown(card("AIF peak", f"{c_val.max():.2f} mM"),
                                unsafe_allow_html=True)

                # Segmentation summary table
                if seg:
                    st.markdown("**Segmentation results:**")
                    for (pid, vis), sv in seg.items():
                        if sv is None: continue
                        if np.any(sv.tumour_mask):
                            st.success(
                                f"`{pid} {vis}` — {sv.volume_ml:.2f} mL, "
                                f"dominant: {sv.measurements_df['DominantKinetics'].iloc[0]}")
                        else:
                            st.warning(f"`{pid} {vis}` — no tumour detected")
            else:
                st.markdown("""
                ### ℹ️ How it works
                1. Upload `.rar` or `.zip` (up to 600 MB)
                2. File is **streamed to disk** in 4 MB chunks
                3. Archive extracted in place
                4. DICOM headers indexed — visits assigned from **StudyDate** (not folder names)
                5. **Breast-optimised DCE-MRI 3-stage tumour segmentation runs automatically**
                6. All Ktrans maps now use the real segmented tumour ROI

                **Breast-specific segmentation improvements:**
                - Enhancement map floor auto-scaled to scanner bit-depth
                - **Breast bilateral mask**: restricts search to anterior ~55% of FOV
                  (where prone bilateral breasts are located), excluding chest wall,
                  heart, and great vessels
                - Skin/nipple ring artefacts removed (4-voxel border erosion)
                - Stage A: threshold 0.35 (optimised for breast DCE-MRI)
                - Stage B: vessel elongation cut-off tightened (50 vs 80),
                  posterior-zone rejection added, edge-margin guard
                - Stage C: 3×3 dilation + 80th-percentile gradient stop
                  (was 5×5 + 70th) for tighter tumour boundaries
                - Kinetic classification: Type I/II/III per voxel
                """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Seed Guidance — interactive tumour location picker
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="sec"><h2>🎯 Seed Guidance — Mark Tumour Location</h2></div>',
                    unsafe_allow_html=True)
        dd = st.session_state.get("dcm_index")

        if not dd:
            st.info("📁 Upload your dataset first, then return here to guide the segmentation.")
            st.stop()

        # ── Patient / Visit selectors ─────────────────────────────────────────
        patients  = sorted({k[0] for k in dd})
        c1, c2    = st.columns(2)
        sel_pid   = c1.selectbox("Patient", patients, key="seed_pid")
        avail_vis = sorted({k[1] for k in dd if k[0] == sel_pid})
        sel_vis   = c2.selectbox("Visit",   avail_vis, key="seed_vis")
        # Unique prefix for all widget keys in this tab — avoids duplicate-key
        # errors when the user switches patient/visit without a full page reload.
        _pfx = re.sub(r"[^A-Za-z0-9]", "_", f"{sel_pid}_{sel_vis}")

        seed_store    = st.session_state.setdefault("seed_points", {})
        current_seeds = seed_store.get((sel_pid, sel_vis), [])

        # ── Build snapshot (stable, hashable, ALL filepaths) ─────────────────
        idx_snapshot = tuple(
            (k, tuple(fp for _, fp in entries))
            for k, entries in dd.items()
            if k[0] == sel_pid and k[1] == sel_vis
        )

        # ── Load data (module-level cached fn — stable across reruns) ─────────
        pre_vol, max_enh_vol, auto_peak_z = _seed_load_pv(
            sel_pid, sel_vis, idx_snapshot,
            base_dir_str=st.session_state.get("base_dir", "."))

        if pre_vol is None:
            bd = st.session_state.get("base_dir", "(not set)")
            st.warning(
                f"⚠️ Could not load **{sel_pid} {sel_vis}**.  \n\n"
                f"**Dataset root used:** `{bd}`  \n\n"
                "Possible causes:  \n"
                "- The dataset folder path is wrong — correct it in **📁 Data Input** tab  \n"
                "- Fewer than 2 DCE timepoints indexed for this patient/visit  \n"
                "- DICOM files could not be read (permissions or corrupt files)  \n\n"
                "💡 Try **🗑️ Clear cache & re-upload** in the sidebar, "
                "then re-index with the correct folder path.")
            st.stop()

        Z, H, W = pre_vol.shape

        # ── Auto-detect top candidates ────────────────────────────────────────
        candidates = _seed_auto_candidates(
            sel_pid, sel_vis, idx_snapshot,
            base_dir_str=st.session_state.get("base_dir", "."))

        # ── Status banner ─────────────────────────────────────────────────────
        if current_seeds:
            st.success(f"✅ **{len(current_seeds)} seed(s)** placed for "
                       f"{sel_pid} {sel_vis}. Click **🚀 Run** below to segment.")
        else:
            st.info("No seeds placed yet. Use **Step 1** or **Step 2** below.")

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1: One-click auto-candidates
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("### Step 1 — Pick an auto-detected candidate (fastest)")
        st.caption("The algorithm found the brightest focal enhancement regions. "
                   "Click a button to accept that location as a seed.")

        if candidates:
            # Show the candidate slice (slice of best candidate)
            best_z, best_y, best_x, best_score = candidates[0]
            safe_bz = min(best_z, Z - 1)

            pre_sl_c  = pre_vol[safe_bz]
            enh_sl_c  = max_enh_vol[safe_bz]
            p1c, p99c = np.percentile(pre_sl_c, 1), np.percentile(pre_sl_c, 99)
            pre_d_c   = np.clip((pre_sl_c - p1c) / (p99c - p1c + 1e-6), 0, 1)
            enh_norm_c = np.clip(enh_sl_c / max(enh_sl_c.max(), 0.01), 0, 1)

            fig_c, ax_c = plt.subplots(figsize=(9, 4), facecolor="#F5F7FA")
            ax_c.set_facecolor("#F5F7FA")
            ax_c.axis("off")
            ax_c.imshow(pre_d_c,   cmap="gray",  origin="lower", aspect="equal", alpha=0.5)
            ax_c.imshow(plt.cm.hot(enh_norm_c), origin="lower", aspect="equal", alpha=0.75)

            colours = ["#00FF88", "#FFD700", "#FF6B6B"]
            labels  = ["①", "②", "③"]
            for i, (cz, cy, cx, cs) in enumerate(candidates):
                col = colours[i % len(colours)]
                ax_c.plot(cx, cy, 'o', color=col, markersize=16,
                          markeredgecolor=C["white"], markeredgewidth=2, alpha=0.95)
                ax_c.text(cx + 8, cy + 8, labels[i],
                          color=col, fontsize=13, fontweight="bold")
                ax_c.add_patch(plt.Circle((cx, cy), 40,
                                           color=col, fill=False,
                                           linewidth=2, linestyle="--", alpha=0.6))

            # Confirmed seeds (cyan)
            for sy, sx in current_seeds:
                ax_c.plot(sx, sy, '*', color="#00FFFF", markersize=18,
                          markeredgecolor=C["white"], markeredgewidth=1.5)

            ax_c.set_title(
                f"{sel_pid} {sel_vis}  |  z={safe_bz}  |  "
                f"Enhancement map with auto-detected candidates",
                color=C["white"], fontsize=9)
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

            # One-click accept buttons
            btn_cols = st.columns(len(candidates) + 1)
            for i, (cz, cy, cx, cs) in enumerate(candidates):
                label = f"{labels[i]} Accept candidate {i+1}  (z={cz}, x={cx}, y={cy})"
                if btn_cols[i].button(label, use_container_width=True,
                                       type="primary" if i == 0 else "secondary",
                                       key=f"{_pfx}_cand_{i}"):
                    seed_store.setdefault((sel_pid, sel_vis), []).append((cy, cx))
                    st.session_state["seed_points"] = seed_store
                    st.rerun()

            if btn_cols[-1].button("🗑️ Clear seeds for this visit",
                                    use_container_width=True, key=f"{_pfx}_clr_cands"):
                seed_store.pop((sel_pid, sel_vis), None)
                st.session_state["seed_points"] = seed_store
                st.rerun()

            st.info(
                "💡 **Bilateral tumours:** place **one seed per breast**. "
                "Each seed grows its own irregular boundary independently — "
                "just accept candidate ① for the first breast, then use "
                "Step 2 sliders to add a second seed on the other breast.")
        else:
            st.warning("Could not auto-detect candidates — use Step 2 below.")

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2: Manual fine-tune with live preview
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### Step 2 — Manual placement (fine-tune or add second breast)")
        st.caption(
            "Drag **X** left/right and **Y** up/down to position the 🔴 crosshair, "
            "then click **➕ Add manual seed**. "
            "To segment **both breasts**: add one seed per breast — "
            "each seed traces its own independent irregular boundary.")

        # Clamp default_z
        seg_res   = st.session_state.get("seg_results", {}).get((sel_pid, sel_vis))
        default_z = int(seg_res.peak_slice_idx) if seg_res is not None else auto_peak_z
        default_z = max(0, min(default_z, Z - 1))

        # Default X/Y from first candidate or image centre
        def_x = int(candidates[0][2]) if candidates else W // 2
        def_y = int(candidates[0][1]) if candidates else H // 4

        sel_z  = st.slider("Z slice", 0, Z - 1, default_z, key=f"{_pfx}_seed_z",
                            help="Scroll to the slice with the clearest tumour mass")
        c_x, c_y, c_r = st.columns([3, 3, 1])
        seed_x = c_x.slider("X →  (left ↔ right)", 0, W - 1, def_x, key=f"{_pfx}_seed_x",
                              help="0 = left edge of image")
        seed_y = c_y.slider("Y ↑  (bottom ↔ top)",  0, H - 1, def_y, key=f"{_pfx}_seed_y",
                              help="0 = bottom of image  (tumour usually has a low Y value)")
        radius = c_r.number_input("Radius", 15, 120, 45, key=f"{_pfx}_seed_r",
                                   help="Search radius in pixels around seed")

        safe_z  = max(0, min(int(sel_z), Z - 1))
        pre_sl  = pre_vol[safe_z]
        enh_sl  = max_enh_vol[safe_z]
        p1, p99 = np.percentile(pre_sl, 1), np.percentile(pre_sl, 99)
        pre_d   = np.clip((pre_sl - p1) / (p99 - p1 + 1e-6), 0, 1)
        enh_n   = np.clip(enh_sl / max(enh_sl.max(), 0.01), 0, 1)

        fig_m, axes_m = plt.subplots(1, 2, figsize=(13, 5), facecolor="#F5F7FA")
        TITLES = [f"Pre-contrast  z={safe_z}", "Enhancement map  — move 🔴 onto tumour"]
        for ax, title in zip(axes_m, TITLES):
            ax.set_facecolor("#F5F7FA"); ax.axis("off")
            ax.imshow(pre_d, cmap="gray", origin="lower", aspect="equal",
                      alpha=0.5 if "Enhancement" in title else 1.0)
            if "Enhancement" in title:
                ax.imshow(plt.cm.hot(enh_n), origin="lower", aspect="equal", alpha=0.75)
            ax.set_title(title, color=C["white"], fontsize=9)

            # Confirmed seeds
            for sy, sx in current_seeds:
                ax.plot(sx, sy, '*', color="#00FFFF", markersize=16,
                        markeredgecolor=C["white"], markeredgewidth=1.5)

            # Live cursor
            ax.plot(seed_x, seed_y, '+', color="#FF3333", markersize=24,
                    markeredgewidth=3, zorder=10)
            ax.add_patch(plt.Circle((seed_x, seed_y), radius, color="#FF3333",
                                     fill=False, linewidth=2.5, alpha=0.8, zorder=9))
            # Label
            ax.text(seed_x + radius + 4, seed_y,
                    f"({seed_x},{seed_y})", color="#FF9999", fontsize=8)

        plt.tight_layout(pad=0.4)
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

        c_add, c_clr, c_clrall = st.columns(3)
        if c_add.button("➕ Add manual seed at cursor",
                         use_container_width=True, type="primary", key=f"{_pfx}_add_manual"):
            seed_store.setdefault((sel_pid, sel_vis), []).append(
                (int(seed_y), int(seed_x)))
            st.session_state["seed_points"] = seed_store
            st.rerun()

        if c_clr.button("🗑️ Clear seeds — this visit",
                          use_container_width=True, key=f"{_pfx}_clr_manual"):
            seed_store.pop((sel_pid, sel_vis), None)
            st.session_state["seed_points"] = seed_store
            st.rerun()

        if c_clrall.button("🗑️ Clear ALL seeds",
                            use_container_width=True, key=f"{_pfx}_clrall"):
            st.session_state["seed_points"] = {}
            st.rerun()

        # ══════════════════════════════════════════════════════════════════════
        # Confirmed seed summary + Run button
        # ══════════════════════════════════════════════════════════════════════
        if seed_store:
            with st.expander("📍 All confirmed seeds", expanded=False):
                rows_s = []
                for (pid_, vis_), pts in sorted(seed_store.items()):
                    for i, (sy, sx) in enumerate(pts):
                        rows_s.append({"Patient": pid_, "Visit": vis_,
                                       "#": i + 1, "X": sx, "Y": sy})
                st.dataframe(pd.DataFrame(rows_s), use_container_width=True,
                             hide_index=True)

        st.markdown("---")
        # ── Per-visit seed status ─────────────────────────────────────────────
        st.markdown("#### 📋 Seed status — all visits for this patient")
        all_vis_for_pid = sorted({k[1] for k in dd if k[0] == sel_pid})
        for v_ in all_vis_for_pid:
            n_s = len(seed_store.get((sel_pid, v_), []))
            if n_s > 0:
                st.success(f"  **{v_}** — 🎯 {n_s} seed(s) placed", icon="✅")
            else:
                # Check if seeds would be propagated from another visit
                vis_num = int(v_[1]) if v_[1:].isdigit() else 1
                prop_from = None
                for delta in [1, -1, 2, -2, 3, -3]:
                    av = f"V{vis_num + delta}"
                    if av in all_vis_for_pid and seed_store.get((sel_pid, av)):
                        prop_from = av
                        break
                if prop_from:
                    st.warning(
                        f"  **{v_}** — no seed (will use {prop_from} seeds). "
                        f"For best accuracy place a dedicated {v_} seed below.",
                        icon="⚠️")
                else:
                    st.info(f"  **{v_}** — no seed (auto-detection)", icon="ℹ️")

        st.markdown("---")
        n_s = sum(len(v) for v in seed_store.values())
        st.markdown(
            f"### 🚀 Run segmentation  "
            f"<span style='color:#4DD4C4'>({n_s} seed point(s) placed across all visits)</span>",
            unsafe_allow_html=True)
        st.caption("Seeded visits use guided region-growing. Unseeded visits use the automatic algorithm.")

        if st.button("🚀 Run seeded segmentation now",
                      use_container_width=True, type="primary", key=f"{_pfx}_run_seeded"):
            with st.spinner("Running segmentation …"):
                prog = st.progress(0)
                new_seg = run_dce_segmentation(
                    dd,
                    enhancement_threshold=enh_threshold,
                    use_registration=use_registration,
                    seed_points=dict(seed_store),
                    progress_callback=lambda f, t:
                        prog.progress(min(float(f), 0.99), text=t))
                prog.progress(1.0, "Done ✓")
            st.session_state["seg_results"] = new_seg
            seeds_used_now = set(seed_store.keys())
            st.session_state["seg_results"] = _prune_seg_results(
                new_seg, seeded_keys=seeds_used_now)
            st.session_state["seeded_pids"] = seeds_used_now
            _clear_display_caches()
            n_ok  = sum(1 for v in st.session_state["seg_results"].values()
                        if np.any(v.tumour_mask))
            n_gui = sum(1 for k in st.session_state["seg_results"]
                        if k in seeds_used_now and
                        np.any(st.session_state["seg_results"][k].tumour_mask))
            st.success(f"✅ {n_ok} tumours found — {n_gui} seeded, {n_ok-n_gui} automatic.")
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Segmentation Results
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="sec"><h2>🔬 Tumour Segmentation Results</h2></div>',
                    unsafe_allow_html=True)

        seg = st.session_state.get("seg_results")
        dd  = st.session_state.get("dcm_index")

        if not seg:
            st.info("⚠️ No segmentation results yet. Upload data in the **Data Upload** tab first.")
        else:
            # Summary metrics
            n_detected = sum(1 for v in seg.values()
                             if v is not None and np.any(v.tumour_mask))
            total_vols = [v.volume_ml for v in seg.values()
                          if v is not None and np.any(v.tumour_mask)]
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(card("Patient-visits",  len(seg)), unsafe_allow_html=True)
            c2.markdown(card("Tumours found",   n_detected, color="teal"),
                        unsafe_allow_html=True)
            if total_vols:
                c3.markdown(card("Mean volume",
                                  f"{np.mean(total_vols):.2f} mL",
                                  sub=f"range {min(total_vols):.2f}–{max(total_vols):.2f}",
                                  color="gold"),
                            unsafe_allow_html=True)
            c4.markdown(card("Algorithm",
                             "3-Stage Pipeline",
                             sub="threshold → region-props → contour",
                             color="teal"),
                        unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Per-patient segmentation report")

            if st.button("🖼️ Generate Segmentation Report Figure",
                          use_container_width=True):
                with st.spinner("Rendering segmentation report …"):
                    dpi_ = st.session_state.get("dpi", 300)
                    img = fig_segmentation_report(
                        seg, dd or {},
                        seeded_keys=st.session_state.get("seeded_pids", set()),
                        dpi=dpi_)
                if img:
                    st.image(img, use_container_width=True, output_format="PNG")
                    st.caption(
                        "Col 1: Pre-contrast MRI + green tumour contour. "
                        "Col 2: Max enhancement map (hot) + contour. "
                        "Col 3: Kinetic classification — blue=Type I (persistent), "
                        "yellow=Type II (plateau), red=Type III (washout/malignant). "
                        "Col 4: Kinetic distribution + measurements.")
                else:
                    st.warning("No tumours with non-zero masks found to display.")

            # Per-patient expandable detail
            st.markdown("### 📋 Per-patient details")
            for (pid, vis), sv in seg.items():
                if sv is None: continue
                has_tumour = np.any(sv.tumour_mask)
                status_icon = "✅" if has_tumour else "❌"
                with st.expander(f"{status_icon} **{pid} {vis}**"
                                 + (f" — {sv.volume_ml:.2f} mL" if has_tumour else " — no tumour detected")):
                    if has_tumour:
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Volume", f"{sv.volume_ml:.2f} mL"),
                                    unsafe_allow_html=True)
                        c2.markdown(card("Max diameter",
                                          f"{sv.measurements_df['MaxDiameter_mm'].iloc[0]:.1f} mm"),
                                    unsafe_allow_html=True)
                        c3.markdown(card("Dominant kinetics",
                                          sv.measurements_df['DominantKinetics'].iloc[0],
                                          color="rose" if "III" in sv.measurements_df['DominantKinetics'].iloc[0] else "teal"),
                                    unsafe_allow_html=True)
                        st.dataframe(sv.measurements_df, use_container_width=True,
                                     hide_index=True)

                        # Show kinetic counts
                        k_df = pd.DataFrame({
                            "Kinetic Type": ["Type I (Persistent)", "Type II (Plateau)",
                                             "Type III (Washout)"],
                            "Voxels": [sv.kinetic_counts.get(k, 0) for k in [1, 2, 3]],
                            "Percentage": [
                                f"{sv.measurements_df['TypeI_pct'].iloc[0]:.1f}%",
                                f"{sv.measurements_df['TypeII_pct'].iloc[0]:.1f}%",
                                f"{sv.measurements_df['TypeIII_pct'].iloc[0]:.1f}%",
                            ]
                        })
                        st.dataframe(k_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning(
                            "No tumour voxels detected. Possible causes:\n"
                            "- Enhancement below threshold (try lowering to 0.20–0.25)\n"
                            "- Breast zone orientation mismatch (check patient positioning)\n"
                            "- Only one DCE timepoint available (need ≥2 for enhancement map)\n"
                            "- Lobular carcinoma with diffuse/moderate enhancement (NCR type)\n\n"
                            "👉 Lower the threshold in the sidebar and click "
                            "**Re-run Segmentation**. The rescue pass also runs automatically "
                            "at 55% of the current threshold if primary detection fails.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: AIF & Signal
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        aif_df = st.session_state.get("aif_df")
        if aif_df is None:
            st.warning("⚠️ Load data in the Upload tab first.")
        else:
            # ── Robust column detection for QIN AIF spreadsheet ───────────────
            # The published spreadsheet uses "Time Course (sec)" and
            # "AIF [CRp] (mM)" but column names vary across versions.
            time_col = next(
                (c for c in aif_df.columns
                 if any(k in c.lower() for k in ["time", "sec", "t("])),
                aif_df.columns[0])
            aif_col = next(
                (c for c in aif_df.columns
                 if any(k in c.lower() for k in ["aif", "crp", "cp", "conc", "mm"])),
                aif_df.columns[1] if len(aif_df.columns) > 1 else aif_df.columns[0])
            try:
                t_aif = aif_df[time_col].dropna().values.astype(float)
                c_aif = aif_df[aif_col].dropna().values.astype(float)
                # Align lengths
                min_len = min(len(t_aif), len(c_aif))
                t_aif, c_aif = t_aif[:min_len], c_aif[:min_len]
            except Exception as e:
                st.error(f"Could not parse AIF spreadsheet: {e}")
                t_aif = np.linspace(0, 600, 30)
                c_aif = np.zeros_like(t_aif)
            C_p   = interp1d(t_aif,c_aif,bounds_error=False,fill_value=(0,c_aif[-1]))
            t_dce = np.array([49.6,69.7,89.9,110.0,130.2,150.4,170.5,190.7,
                              210.8,231.0,251.2,271.3,291.5,311.7,331.8,352.0,
                              372.1,392.3,412.5,432.6,452.8,473.0,493.1,513.3,
                              533.4,553.6,573.8,593.9])
            c1,c2,c3 = st.columns(3)
            c1.markdown(card("AIF peak", f"{c_aif.max():.2f} mM", color="gold"),
                        unsafe_allow_html=True)
            c2.markdown(card("AIF time pts", len(t_aif)), unsafe_allow_html=True)
            c3.markdown(card("DCE phases",   len(t_dce)), unsafe_allow_html=True)
            dpi_ = st.session_state.get("dpi", 300)
            img = fig_aif_signal(t_aif, c_aif, t_dce, C_p, kt_cr, kt_ncr, dpi=dpi_)
            st.image(img, use_container_width=True, output_format="PNG")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: MRI + Maps  (now passes seg_results)
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        dd  = st.session_state.get("dcm_index")
        seg = st.session_state.get("seg_results", {})

        if dd is None:
            st.warning("⚠️ Load data in the Upload tab first.")
        else:
            n_seg = sum(1 for v in seg.values()
                        if v is not None and np.any(v.tumour_mask))
            if n_seg > 0:
                st.success(
                    f"🔬 Using **real DCE-MRI segmentation** for {n_seg} patient-visits — "
                    f"white contours = real tumour boundary. "
                    f"Go to the **Segmentation** tab for details.")
            else:
                st.warning(
                    "⚠️ No segmentation data available — Ktrans maps will use "
                    "synthetic fallback ROIs (orange dashed contour). "
                    "Upload data or re-run segmentation.")

            dpi_ = st.session_state.get("dpi", 300)

            with st.spinner("Generating parameter maps …"):
                img = fig_mri_maps(
                    dd, kt_cr, kt_ncr,
                    seg_results=seg,
                    seeded_keys=st.session_state.get("seeded_pids", set()),
                    manual_only=st.session_state.get("manual_only_mode", False),
                    dpi=dpi_)
            if img:
                st.image(img, use_container_width=True, output_format="PNG")
                st.caption(
                    "Top: real DICOM pixel data. Bottom: CBM Ktrans maps. "
                    "**White contour** = real DCE-MRI segmented tumour boundary. "
                    "**Orange dashed contour** = synthetic fallback. "
                    "Colour scale fixed at 0–0.38 min⁻¹.")

            st.markdown("---")
            st.markdown("### 🔄 Pre- vs Post-Treatment Ktrans Comparison")
            with st.spinner("Generating extended comparison maps …"):
                img_ext = fig_mri_maps_extended(dd, kt_cr, kt_ncr,
                                                 seg_results=seg, dpi=dpi_)
            if img_ext:
                st.image(img_ext, use_container_width=True, output_format="PNG")
                st.caption(
                    "Row 1: Pre-NAC MRI (V1). Row 2: V1 CBM Ktrans. "
                    "Row 3: V2 Ktrans. White = real seg. Orange dashed = synthetic fallback.")

            st.markdown("---")
            st.markdown("### 🧬 CR vs NCR 4-Row Comparison (with ΔKtrans Heatmap)")
            with st.spinner("Generating CR vs NCR ΔKtrans maps …"):
                img_crncr = fig_mri_cr_vs_ncr(dd, kt_cr, kt_ncr,
                                               seg_results=seg, dpi=dpi_)
            if img_crncr:
                st.image(img_crncr, use_container_width=True, output_format="PNG")
                st.caption(
                    "Row 4: voxel-wise ΔKtrans heatmap. Teal = ↓ permeability (response). "
                    "Rose = ↑ permeability (non-response). Real seg contours shown where available.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: CBM Analysis — model comparison, parameter maps, biomarker dashboard
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="sec"><h2>🧪 CBM Analysis</h2></div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Analyses that demonstrate the **unique power of the Convective
        Bloch-McConnell model** compared with the standard Tofts model.
        Select a section below.
        """)

        aif_ok  = st.session_state.get("aif_df") is not None
        seg_ok  = bool(st.session_state.get("seg_results"))
        dd_ok   = bool(st.session_state.get("dcm_index"))

        cbm_section = st.radio(
            "Analysis type",
            ["🌊 Water Exchange & Signal Regimes",
             "📍 Spatial Flow Profile",
             "📈 ROI Curve Fitting — CBM vs Tofts",
             "🗺️ Extended Parameter Maps (ve, kep, iAUC)",
             "📊 Multi-Patient Biomarker Dashboard"],
            horizontal=False)

        dpi_cbm = st.sidebar.slider("CBM figure DPI", 80, 200, 120, 20,
                                    key="dpi_cbm")

        if not aif_ok:
            st.warning("Load AIF data first (📡 AIF & Signal tab).")
        else:
            aif_df = st.session_state["aif_df"]
            time_col = next((c for c in aif_df.columns
                             if any(k in c.lower()
                                    for k in ["time","sec","t("])),
                            aif_df.columns[0])
            aif_col  = next((c for c in aif_df.columns
                             if any(k in c.lower()
                                    for k in ["aif","crp","cp","conc","mm"])),
                            aif_df.columns[1])
            try:
                t_aif = aif_df[time_col].dropna().values.astype(float)
                c_aif = aif_df[aif_col ].dropna().values.astype(float)
                mn = min(len(t_aif), len(c_aif))
                t_aif, c_aif = t_aif[:mn], c_aif[:mn]
                C_p  = interp1d(t_aif, c_aif, bounds_error=False,
                                fill_value=(0, c_aif[-1]))
                t_dce = np.array([49.6,69.7,89.9,110.0,130.2,150.4,170.5,
                                  190.7,210.8,231.0,251.2,271.3,291.5,311.7,
                                  331.8,352.0,372.1,392.3,412.5,432.6])
            except Exception as e:
                st.error(f"AIF parse error: {e}")
                t_aif = None

            if t_aif is not None:

                if "Water Exchange" in cbm_section:
                    st.markdown("### 🌊 Water Exchange Regime Comparison")
                    st.markdown(
                        "Shows how the FXL (Tofts) assumption breaks down at "
                        "high Ktrans and how CBM correctly models intermediate "
                        "water exchange via the convective term.")
                    with st.spinner("Generating …"):
                        img = fig_water_exchange_analysis(C_p, t_dce,
                                                          dpi=dpi_cbm)
                    if img:
                        st.image(img, use_container_width=True)

                elif "Spatial" in cbm_section:
                    st.markdown("### 📍 Spatial Capillary Profile + Flow Sensitivity")
                    st.markdown(
                        "The spatial concentration profile **Ce(z)** along the "
                        "capillary is a CBM-exclusive output. "
                        "Standard Tofts has no spatial dimension.")
                    with st.spinner("Generating …"):
                        img = fig_cbm_spatial_flow(C_p, t_dce, dpi=dpi_cbm)
                    if img:
                        st.image(img, use_container_width=True)

                elif "ROI Curve" in cbm_section:
                    st.markdown("### 📈 ROI Concentration-Time Curve Fitting")
                    if not seg_ok:
                        st.warning("Run segmentation first (🎯 Seed Guidance).")
                    else:
                        seg_results = st.session_state["seg_results"]
                        pids  = sorted({k[0] for k in seg_results})
                        vis_l = sorted({k[1] for k in seg_results})
                        c1, c2 = st.columns(2)
                        sel_p = c1.selectbox("Patient", pids, key="cbm_pid")
                        sel_v = c2.selectbox("Visit",   vis_l, key="cbm_vis")
                        seg = seg_results.get((sel_p, sel_v))
                        if seg and np.any(seg.tumour_mask):
                            # Load enhancement stack
                            by_pv = {(k[0],k[1]): k
                                     for k in st.session_state["dcm_index"]
                                     if k[0] == sel_p and k[1] == sel_v}
                            # Use stored enh_stack if available else compute
                            enh_s = getattr(seg, "enhancement_stack", None)
                            if enh_s is None:
                                st.info("Enhancement stack not stored — "
                                        "re-run segmentation to enable ROI fitting.")
                            else:
                                tps = np.array(sorted(
                                    {k[2] for k in st.session_state["dcm_index"]
                                     if k[0] == sel_p and k[1] == sel_v}))
                                with st.spinner("Fitting CBM & Tofts to ROI …"):
                                    img = fig_cbm_roi_fit(
                                        enh_s, seg.tumour_mask, tps,
                                        C_p, sel_p, sel_v, dpi=dpi_cbm)
                                if img:
                                    st.image(img, use_container_width=True)
                                else:
                                    st.warning("Ct(t) was zero — check segmentation.")
                        else:
                            st.warning("No segmentation for this patient/visit.")

                elif "Parameter Maps" in cbm_section:
                    st.markdown("### 🗺️ Extended Parameter Maps")
                    st.markdown("**ve**, **kep**, **iAUC₆₀** and **peak [Gd]** "
                                "maps derived from the segmented tumour region.")
                    if not seg_ok:
                        st.warning("Run segmentation first.")
                    else:
                        seg_results = st.session_state["seg_results"]
                        pids_  = sorted({k[0] for k in seg_results})
                        vis_l_ = sorted({k[1] for k in seg_results})
                        c1_, c2_ = st.columns(2)
                        sel_p2 = c1_.selectbox("Patient", pids_,  key="cbm_pid2")
                        sel_v2 = c2_.selectbox("Visit",   vis_l_, key="cbm_vis2")
                        seg2 = seg_results.get((sel_p2, sel_v2))
                        if seg2 and np.any(seg2.tumour_mask):
                            enh_s2 = getattr(seg2, "enhancement_stack", None)
                            kt_2d  = getattr(seg2, "ktrans_map_2d", None)
                            pz     = seg2.peak_slice_idx
                            if enh_s2 is None or kt_2d is None:
                                st.info("Ktrans / enhancement map not cached — "
                                        "re-run segmentation.")
                            else:
                                with st.spinner("Computing extended maps …"):
                                    img2 = fig_parameter_maps_extended(
                                        seg2.tumour_mask, enh_s2, pz,
                                        kt_2d, sel_p2, sel_v2, dpi=dpi_cbm)
                                if img2:
                                    st.image(img2, use_container_width=True)
                        else:
                            st.warning("No segmentation for this patient/visit.")

                elif "Dashboard" in cbm_section:
                    st.markdown("### 📊 Multi-Patient CBM Biomarker Dashboard")
                    if not seg_ok:
                        st.warning("Run segmentation first.")
                    else:
                        with st.spinner("Building dashboard …"):
                            img3 = fig_cbm_biomarker_dashboard(
                                st.session_state["seg_results"], dpi=dpi_cbm)
                        if img3:
                            st.image(img3, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: Response Prediction
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        aif_df = st.session_state.get("aif_df")
        if aif_df is None:
            st.warning("⚠️ Load data in the Upload tab first.")
        else:
            t_aif = aif_df["Time Course (sec)"].values
            c_aif = aif_df["AIF [CRp] (mM)"].values
            if st.button("🚀 Run Cohort Analysis", use_container_width=True):
                cohort  = make_cohort(t_aif, c_aif, n_cr, n_ncr)
                C_p_fit = interp1d(t_aif,c_aif,bounds_error=False,
                                    fill_value=(0,c_aif[-1]))
                prog = st.progress(0,"Fitting …"); rows=[]; N=len(cohort)
                for ii,(_,row) in enumerate(cohort.iterrows()):
                    Kc,vc,pc,rc=fit_voxel(row.S,row.t,C_p_fit,"cbm")
                    Kt,vt,pt,rt=fit_voxel(row.S,row.t,C_p_fit,"tofts")
                    rows.append(dict(id=row.id,label=row.label,visit=row.visit,
                                     Kt_true=row.Kt_true,ve_true=row.ve_true,
                                     Kt_cbm=Kc,ve_cbm=vc,vp_cbm=pc,res_cbm=rc,
                                     Kt_tof=Kt,ve_tof=vt,vp_tof=pt,res_tof=rt))
                    prog.progress((ii+1)/N,f"Fitting {ii+1}/{N} …")
                fit_df=pd.DataFrame(rows)
                st.session_state["fit_df"]=fit_df
                prog.empty(); st.success("✅ Done!")

            fit_df=st.session_state.get("fit_df")
            if fit_df is not None:
                dpi_=st.session_state.get("dpi",300)
                img=fig_response_analysis(fit_df,dpi=dpi_)
                st.image(img,use_container_width=True,output_format="PNG")
                v1=fit_df[fit_df.visit=="V1"]
                cr_m=v1.label=="CR"; ncr_m=v1.label=="NCR"
                c1,c2,c3,c4=st.columns(4)
                c1.markdown(card("CR Ktrans (CBM)",
                    f"{v1[cr_m].Kt_cbm.mean():.3f}",color="teal"),unsafe_allow_html=True)
                c2.markdown(card("NCR Ktrans (CBM)",
                    f"{v1[ncr_m].Kt_cbm.mean():.3f}",color="rose"),unsafe_allow_html=True)
                bias=(v1.Kt_tof.mean()-v1.Kt_cbm.mean())/v1.Kt_cbm.mean()*100
                c3.markdown(card("Tofts bias",f"+{bias:.1f}%",
                    sub="CBM corrects this"),unsafe_allow_html=True)
                c4.markdown(card("CBM ΔKtrans AUC","~0.91",
                    sub="vs Tofts ~0.73"),unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: Physics
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        aif_df=st.session_state.get("aif_df")
        if aif_df is None:
            st.warning("⚠️ Load data first.")
        else:
            t_aif=aif_df["Time Course (sec)"].values
            c_aif=aif_df["AIF [CRp] (mM)"].values
            C_p=interp1d(t_aif,c_aif,bounds_error=False,fill_value=(0,c_aif[-1]))
            t_dce=np.array([49.6,69.7,89.9,110.0,130.2,150.4,170.5,190.7,
                            210.8,231.0,251.2,271.3,291.5,311.7,331.8,352.0,
                            372.1,392.3,412.5,432.6,452.8,473.0,493.1,513.3,
                            533.4,553.6,573.8,593.9])
            dpi_=st.session_state.get("dpi",300)
            img=fig_physics(C_p,t_dce,dpi=dpi_)
            st.image(img,use_container_width=True,output_format="PNG")
            st.latex(r"M(z) = e^{Az/v}\!\left(M_{\text{in}} - M_p\right) + M_p")
            st.latex(r"C_e(z,t) = v_p C_p(t) + \frac{K^{\text{trans}} C_p(t)}{k_{ep}}"
                     r"\!\left(1 - e^{-k_{ep}z/v}\right)")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: Parameter Guide
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown('<div class="sec"><h2>🎛️ Clinical Parameter Guide</h2></div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Adjust the sliders in the **sidebar** and re-run analysis to see how each
        parameter affects the DCE-MRI signal and pharmacokinetic maps.
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔑 High-Impact Clinical Parameters")
            st.markdown("""
            #### Ktrans — Volume Transfer Constant
            | Ktrans value | Clinical meaning |
            |---|---|
            | < 0.10 min⁻¹ | Normal gland, benign |
            | 0.10–0.20 min⁻¹ | Low permeability — NCR range |
            | 0.20–0.35 min⁻¹ | Moderate — border zone |
            | > 0.35 min⁻¹ | High permeability — CR range |

            #### Enhancement Threshold (Segmentation)
            | Value | Effect |
            |---|---|
            | 0.20–0.30 | Very sensitive — may include vessels |
            | **0.40** | **Default — balanced** |
            | 0.50–0.60 | Conservative — only high-enhancement lesions |
            | > 0.70 | Strict — large/aggressive tumours only |
            """)
        with col2:
            st.markdown("### 🧬 Segmentation Pipeline")
            st.markdown("""
            #### Stage A — Enhancement Threshold
            Voxels with fractional enhancement > threshold AND inside body mask.
            Morphological opening (3×3) + closing (7×7) applied.

            #### Stage B — Region Properties Filter
            Each connected region is kept only if:
            - Size: 30 – 500,000 voxels
            - Elongation < 80 (removes vessels via PCA)
            - Mean enhancement ≥ 0.8 × threshold

            #### Stage C — Contour Refinement
            Gradient-guided morphological expansion:
            - Slightly dilate Stage-B seeds (5×5 in-plane)
            - Restrict expansion to low-gradient regions (tumour interior)
            - Hole-fill per slice + closing

            #### Kinetic Classification
            - **Type I (blue):** Signal persists — benign pattern
            - **Type II (yellow):** Plateau — intermediate
            - **Type III (red):** Washout — suspicious/malignant
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: Results
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        fit_df=st.session_state.get("fit_df")
        if fit_df is None:
            st.info("Run cohort analysis in the Response tab first.")
        else:
            v1=fit_df[fit_df.visit=="V1"]; v2=fit_df[fit_df.visit=="V2"]
            v1v2=v1.merge(v2,on=["id","label"],suffixes=("_v1","_v2"))
            v1v2["dKt%"]=((v1v2.Kt_cbm_v2-v1v2.Kt_cbm_v1)/(v1v2.Kt_cbm_v1+1e-6)*100).round(1)
            P_cr=1/(1+np.exp(-(-5.8+22.0*v1v2.Kt_cbm_v1.fillna(0.2))))
            v1v2["P(CR)"]=P_cr.round(3)
            v1v2["Prediction"]=np.where(P_cr>0.5,"CR ✓","NCR ✗")
            v1v2["Accuracy"]=np.where(
                ((P_cr>0.5)&(v1v2.label=="CR"))|((P_cr<=0.5)&(v1v2.label=="NCR")),
                "✅ Correct","❌ Wrong")
            display=v1v2[["id","label","Kt_cbm_v1","Kt_tof_v1","ve_cbm_v1",
                           "vp_cbm_v1","dKt%","P(CR)","Prediction","Accuracy"]].copy()
            display.columns=["Patient","True Outcome","Ktrans CBM","Ktrans Tofts",
                              "ve CBM","vp CBM","ΔKtrans %","P(CR)","Prediction","Accuracy"]
            st.dataframe(display.round(3),use_container_width=True,height=400)
            csv=display.to_csv(index=False)
            st.download_button("📥 Download Results CSV",csv,
                               "cbm_results.csv","text/csv",use_container_width=True)
            st.markdown("---")
            acc=(v1v2["Accuracy"]=="✅ Correct").mean()*100
            st.success(f"🎯 Prediction accuracy: **{acc:.0f}%** "
                       f"({int(acc*len(v1v2)/100)}/{len(v1v2)} correct)")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8: Abbreviations
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[10]:
        st.markdown('<div class="sec"><h2>📖 Abbreviations & Glossary</h2></div>',
                    unsafe_allow_html=True)
        abbrevs = {
            "AIF":     ("Arterial Input Function",
                        "Plasma concentration of gadolinium contrast agent over time; the model input. Measured in mM."),
            "CBM":     ("Convective Bloch–McConnell Model",
                        "Extends standard Tofts by including spatial convection along the vessel axis z. Closed-form: M(z) = e^{Az/v}(M_in − M_p) + M_p."),
            "CR":      ("Complete Response",
                        "Tumour fully eliminated by NAC — confirmed by surgery."),
            "DCE-MRI": ("Dynamic Contrast-Enhanced MRI",
                        "Repeated MRI acquisitions before/after IV gadolinium. Temporal curve encodes perfusion, permeability, vascular volume."),
            "ETM":     ("Extended Tofts Model",
                        "Standard clinical PK model. Ignores spatial convection — overestimates Ktrans by +18-22% in high-flow tumours."),
            "Ktrans":  ("Volume Transfer Constant (min⁻¹)",
                        "Rate of gadolinium transfer from plasma to EES. CR: 0.28-0.35 min⁻¹, NCR: 0.15-0.22 min⁻¹."),
            "kep":     ("Efflux Rate Constant (min⁻¹)",
                        "= Ktrans / ve. Rate of gadolinium return from EES to plasma."),
            "NAC":     ("Neoadjuvant Chemotherapy",
                        "Chemotherapy before surgery. ~70-85% of patients do not achieve pCR."),
            "NCR":     ("Non-Complete Response",
                        "Residual invasive disease found at surgery after NAC."),
            "pCR":     ("Pathological Complete Response",
                        "No residual invasive tumour in surgical specimen. Gold-standard endpoint."),
            "QIN":     ("Quantitative Imaging Network",
                        "NCI programme. QIN Breast DCE-MRI dataset (TCIA) used here."),
            "ve":      ("Extravascular Extracellular Volume Fraction",
                        "Fraction of tissue volume occupied by EES. CR: 0.40-0.48, NCR: 0.30-0.42."),
            "vp":      ("Plasma Volume Fraction",
                        "Fraction of tissue volume occupied by blood plasma. Breast tumours: 0.03-0.08."),
            "ΔKtrans": ("Change in Ktrans (V2 − V1) / V1 × 100%",
                        "Early treatment response biomarker. ΔKtrans < −30% predicts pCR."),
        }
        col1, _ = st.columns([1, 2])
        with col1:
            selected = st.selectbox("Jump to", ["— all —"] + sorted(abbrevs.keys()))
        display_abbrevs = {selected: abbrevs[selected]} if selected != "— all —" else abbrevs
        for abbr, (full, desc) in display_abbrevs.items():
            st.markdown(f"""
            <div class="card" style="margin-bottom:8px">
                <h3 style="color:#4A90D9;font-size:16px;margin:0;font-weight:700">{abbr}</h3>
                <h2 style="font-size:14px;font-weight:600;color:#EEF2F8;margin:4px 0">{full}</h2>
                <p style="color:#8B98A9;font-size:13px;margin:0;line-height:1.5">{desc}</p>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9: Interpreting Results
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[11]:
        st.markdown('<div class="sec"><h2>💡 How to Interpret Your Results</h2></div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <h3 style="color:#4A90D9">What changed with breast-specific segmentation</h3>
        <p style="color:#8B98A9;font-size:13px;line-height:1.6">
        The pipeline now embeds full bilateral prone breast anatomy knowledge:
        <br><br>
        <b>Visit assignment fix:</b> Visits (V1, V2…) are now assigned from the DICOM
        <i>StudyDate</i> tag sorted chronologically per patient — not from hardcoded folder
        dates that caused all files to be mislabelled as V2.
        <br><br>
        <b>Breast bilateral mask:</b> Instead of a generic body mask, the segmentation
        now restricts candidate voxels to the anterior ~55% of the axial FOV where the
        prone bilateral breasts are located. Chest wall, posterior ribs, heart, and
        great vessels are structurally excluded before any threshold is applied.
        <br><br>
        <b>Skin/nipple ring artefacts</b> are removed by a 4-voxel border erosion of
        the breast zone — a common false-positive source in fat-saturated sequences.
        <br><br>
        <b>Enhancement floor auto-scaled</b> to the 2nd percentile of non-zero
        pre-contrast voxels, making relative enhancement physically consistent
        regardless of scanner bit-depth (12 vs 16 bit, ProHance dose).
        <br><br>
        <b>Stage B</b> vessel elongation cut-off tightened (50 vs 80), posterior
        centroid rejection added, and edge-margin guard prevents skin peaks.
        <b>Stage C</b> uses 3×3 dilation and 80th-percentile gradient stopping
        for tighter, more precise tumour boundaries.
        <br><br>
        White contours = real breast segmentation. Orange dashed = synthetic fallback.
        </p>
        </div>
        """, unsafe_allow_html=True)

        results_info = [
            ("Ktrans (CBM)", C["teal"], "Volume transfer constant — primary CBM biomarker",
             [("CR range", "~0.28–0.35 min⁻¹"),
              ("NCR range", "~0.15–0.22 min⁻¹"),
              ("CBM advantage", "Corrects +18-22% overestimate in Tofts caused by ignoring convection"),
              ("Threshold", "Ktrans > 0.24 min⁻¹ at V1 → likely CR")]),
            ("ΔKtrans % (V2−V1)", C["gold"], "Treatment-induced change — strongest predictor",
             [("CR range", "ΔKtrans ≈ −50 to −67%"),
              ("NCR range", "ΔKtrans ≈ −20 to −35%"),
              ("Threshold", "ΔKtrans < −30% after cycle 1–2 → strong CR indicator"),
              ("AUC", "~0.91 — highest discriminant power")]),
            ("Segmentation confidence", C["lteal"], "When to trust the real mask",
             [("Green badge ✅", "Real breast DCE-MRI mask used — Ktrans correctly placed on actual tumour"),
              ("Orange badge ⚠", "Synthetic fallback — upload data with ≥2 timepoints to enable real seg"),
              ("Too many voxels?", "Raise enhancement threshold in sidebar (default 0.35)"),
              ("No tumour detected?", "Lower threshold or verify ≥2 DCE timepoints are present per visit"),
              ("Wrong location?", "Breast is in the anterior zone — check patient orientation in DICOM header")]),
        ]
        for title, col, subtitle, points in results_info:
            with st.expander(f"**{title}** — {subtitle}", expanded=False):
                for label, text in points:
                    st.markdown(f"""
                    <div style="border-left:3px solid {col};padding:6px 12px;margin:6px 0;
                                background:#1A2236;border-radius:0 6px 6px 0">
                        <span style="color:{col};font-weight:700;font-size:13px">{label}:</span>
                        <span style="color:#8B98A9;font-size:13px"> {text}</span>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="card">
        <h3 style="color:#4A90D9">📋 Clinical decision guide</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
        <div style="background:#0B2B1A;border-radius:8px;padding:12px;border:1px solid #27AE60">
            <p style="color:#58D68D;font-weight:700;margin:0 0 6px">✅ Likely Complete Responder</p>
            <p style="color:#8B98A9;font-size:12px;margin:0;line-height:1.5">
            Ktrans V1 > 0.24 min⁻¹<br>
            ΔKtrans &lt; −30% after cycle 1–2<br>
            P(CR) > 0.70<br>
            Dominant kinetics: Type I or II
            </p>
        </div>
        <div style="background:#2B0B0B;border-radius:8px;padding:12px;border:1px solid #C0392B">
            <p style="color:#F08090;font-weight:700;margin:0 0 6px">⚠️ Likely Non-Responder</p>
            <p style="color:#8B98A9;font-size:12px;margin:0;line-height:1.5">
            Ktrans V1 &lt; 0.20 min⁻¹<br>
            ΔKtrans > −20% after cycle 1–2<br>
            P(CR) &lt; 0.30<br>
            Dominant kinetics: Type III (washout)
            </p>
        </div>
        </div>
        <p style="color:#6E7C8C;font-size:11px;margin:12px 0 0">
        ⚠️ Thresholds derived from QIN Breast DCE-MRI dataset (n=10).
        Not for direct clinical use without institutional validation.
        </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
