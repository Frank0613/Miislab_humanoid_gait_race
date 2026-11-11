# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Name labels floating above the racing robots.

Every label is a pixel-letter mesh written to its own USD file and shown as a visualization-marker prototype, so
it appears in every visualizer (Kit, Newton) and is moved with the robot at every step.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# 5 x 7 pixel letters; lower case is drawn in upper case, unknown characters as "?"
PIXEL_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}

# corners and faces of a unit cube centered on the origin, faces as quads with outward normals
_CUBE_FACES = (
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((-1, 0, 0), ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1))),
    ((0, 1, 0), ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ((0, 0, -1), ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1))),
)


def text_pixels(text: str) -> tuple[list[tuple[int, int]], int]:
    """``(column, row)`` of every lit pixel of ``text``, row 0 at the top, and the number of columns."""
    pixels = []
    for i, letter in enumerate(text.upper()):
        for row, line in enumerate(PIXEL_FONT.get(letter, PIXEL_FONT["?"])):
            pixels += [(6 * i + col, row) for col, on in enumerate(line) if on == "1"]
    return pixels, max(6 * len(text) - 1, 1)


def write_label_usd(path: str, text: str, pixel: float, yaw: float):
    """Write ``text`` as one triangle mesh, centered on the origin and readable when looking along ``yaw``.

    The letters stand in the vertical plane facing the reader, so a label seen from that direction reads left to
    right; they are extruded half a pixel, so they stay visible from the side.
    """
    from pxr import Gf, Usd, UsdGeom, Vt  # noqa: PLC0415

    pixels, columns = text_pixels(text)
    # local frame: the reader looks along +x, so their right is -y, and up is +z
    half = np.array([pixel * 0.25, pixel * 0.505, pixel * 0.505])
    points, normals, indices = [], [], []
    for col, row in pixels:
        center = np.array([0.0, (columns / 2 - col - 0.5) * pixel, (3.5 - row - 0.5) * pixel])
        for normal, corners in _CUBE_FACES:
            base = len(points)
            points += [center + half * np.array(corner) for corner in corners]
            normals += [normal] * 4
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    # turn the reading direction from +x to the yaw
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    points = np.asarray(points) @ rotation.T
    normals = np.asarray(normals, dtype=float) @ rotation.T

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Label")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Label/text")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points.tolist()]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals.tolist()]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    stage.GetRootLayer().Save()


class RaceLabels:
    """One name label per lane, floating above its robot."""

    _count = 0

    def __init__(self, names: list[str], max_width: float, pixel: float, yaw: float, color, height: float):
        """
        Args:
            names: Label of every lane, lane 0 first.
            max_width: Widest label [m]; the letters of long names are made smaller to fit.
            pixel: Size of one letter pixel of a short name [m].
            yaw: Direction the labels are read from, around z [rad].
            color: RGB color of the letters.
            height: Height of the label center above the robot's root [m].
        """
        self.height = height
        self._dir = tempfile.TemporaryDirectory(prefix="race_labels_")
        markers = {}
        for lane, name in enumerate(names):
            path = os.path.join(self._dir.name, f"label_{lane}.usda")
            columns = text_pixels(name)[1]
            write_label_usd(path, name, min(pixel, max_width / columns), yaw)
            markers[f"lane_{lane}"] = sim_utils.UsdFileCfg(
                usd_path=path, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5)
            )
        # a new prim path for every set of labels: marker prototypes cannot be replaced
        RaceLabels._count += 1
        prim_path = f"/Visuals/RaceLabels_{RaceLabels._count}"
        self._markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path, markers=markers))
        self._indices = torch.arange(len(names))

    def hide(self):
        """Hide every label, for good."""
        self._markers.set_visibility(False)

    def update(self, root_pos_w: torch.Tensor):
        """Move every label above its robot."""
        translations = root_pos_w.clone()
        translations[:, 2] += self.height
        self._markers.visualize(translations=translations, marker_indices=self._indices.to(root_pos_w.device))
