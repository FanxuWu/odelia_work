#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from glob import glob
import numpy as np
import torch
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, NormalizeIntensity
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as nd_rotate


def find_case(root, modality):
    pattern1 = os.path.join(root, "**", f"{modality}.nii.gz")
    pattern2 = os.path.join(root, "**", f"{modality}.nii")
    files = sorted(glob(pattern1, recursive=True) + glob(pattern2, recursive=True))
    if not files:
        raise FileNotFoundError(f"No file found for modality '{modality}' under: {root}")
    return files[0]


def load_volume(path):
    pipeline = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),       # (C, H, W, D)
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
    vol = pipeline(path)           # torch.Tensor or numpy -> monai returns numpy by default
    if isinstance(vol, torch.Tensor):
        vol = vol.numpy()
    return vol


def middle_slice_z(vol_c_hwd):
    # vol: (C, H, W, D)
    c, h, w, d = vol_c_hwd.shape
    mid = d // 2
    # use channel 0 by default
    img = vol_c_hwd[0, :, :, mid]
    return img


def flip_3d(vol, axis):
    # axis in {0,1,2} for (H,W,D) in channel-first (C,H,W,D) -> operate on (H,W,D)
    v = vol[0]  # (H, W, D)
    v = np.flip(v, axis=axis)
    out = v[None, ...]  # back to (C,H,W,D)
    return out


def rotate_3d(vol, axis_pair, angle_deg):
    # vol: (C,H,W,D) -> operate on (H,W,D)
    # axis_pair: tuple like (0,1), (0,2), (1,2) corresponding to rotation plane
    v = vol[0]  # (H, W, D)
    # order=1 (bilinear), reshape=False keep same size
    v_rot = nd_rotate(v, angle=angle_deg, axes=axis_pair, reshape=False, order=1, mode="nearest")
    out = v_rot[None, ...]
    return out


def make_grid(images, titles, ncols=5, out_path="figures/augmentation_examples.png"):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(3.8 * ncols, 3.8 * nrows), dpi=120)
    for i, (img, ttl) in enumerate(zip(images, titles), start=1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(img, cmap="gray")
        plt.title(ttl, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Load 3D MRI with MONAI and visualize basic augmentations.")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing NIfTI files")
    parser.add_argument("--modality", type=str, default="T2", help="Modality name to load (e.g., T2, Pre, Post_1)")
    parser.add_argument("--case", type=str, default="", help="Optional direct path to a NIfTI file")
    parser.add_argument("--out", type=str, default="F:\\odelia_work\\figures\\augmentation_examples.png",
                        help="Output figure path")
    args = parser.parse_args()

    if args.case:
        case_path = args.case
        if not os.path.isfile(case_path):
            raise FileNotFoundError(f"Case file not found: {case_path}")
    else:
        case_path = find_case(args.root, args.modality)

    vol = load_volume(case_path)          # (C,H,W,D)
    img_orig = middle_slice_z(vol)

    # flips along H(0), W(1), D(2)
    vol_fx = flip_3d(vol, axis=0)
    vol_fy = flip_3d(vol, axis=1)
    vol_fz = flip_3d(vol, axis=2)

    img_fx = middle_slice_z(vol_fx)
    img_fy = middle_slice_z(vol_fy)
    img_fz = middle_slice_z(vol_fz)

    # rotations ±10° around X, Y, Z:
    # rotate around X -> plane (1,2), around Y -> plane (0,2), around Z -> plane (0,1)
    rxs = [rotate_3d(vol, axis_pair=(1, 2), angle_deg=a) for a in (+10, -10)]
    rys = [rotate_3d(vol, axis_pair=(0, 2), angle_deg=a) for a in (+10, -10)]
    rzs = [rotate_3d(vol, axis_pair=(0, 1), angle_deg=a) for a in (+10, -10)]

    imgs_rot = [middle_slice_z(v) for v in (rxs + rys + rzs)]
    titles_rot = ["RotX +10°", "RotX -10°", "RotY +10°", "RotY -10°", "RotZ +10°", "RotZ -10°"]

    images = [img_orig, img_fx, img_fy, img_fz] + imgs_rot
    titles = ["Original", "Flip X", "Flip Y", "Flip Z"] + titles_rot

    saved = make_grid(images, titles, ncols=5, out_path=args.out)
    print(f"[OK] Saved augmentation figure to: {saved}")
    print(f"[INFO] Case: {case_path}")


if __name__ == "__main__":
    main()
