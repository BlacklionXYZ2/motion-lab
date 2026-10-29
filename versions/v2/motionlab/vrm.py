"""Read a VRM avatar's humanoid skeleton in its rest pose (no Unity needed).

A ``.vrm`` file is binary glTF: a 12-byte header, then a JSON chunk and a binary chunk. Only the
JSON is needed here: the node hierarchy (translation, rotation, scale) and the humanoid bone map,
``VRMC_vrm`` (VRM 1.0) or ``VRM`` (VRM 0.x). glTF's frame is right-handed, +Y up, in metres.

Positions are returned in motion-lab's convention: **she faces +Z and her left is +X**. VRM 1.0
models already do; VRM 0.x models face −Z, so they are turned 180° about +Y (x → −x, z → −z).
The file itself is only read, never changed.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
JSON_CHUNK = 0x4E4F534A  # "JSON"


@dataclass(frozen=True)
class VrmSkeleton:
    bone_nodes: dict[str, int]  # humanoid bone name → glTF node index
    world_positions: dict[str, np.ndarray]  # humanoid bone name → rest position, metres
    parents: dict[str, str | None]  # nearest humanoid ancestor of each humanoid bone
    meta: dict[str, Any]
    vrm_version: str  # "1.0" or "0.x"
    turned: bool  # True when positions were turned 180° to face +Z


def read_gltf_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError(f"{path} is not a binary glTF 2.0 file")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != JSON_CHUNK:
        raise ValueError(f"{path}: the first chunk is not JSON")
    return json.loads(data[20 : 20 + chunk_length])  # type: ignore[no-any-return]


def _quat_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = q  # glTF stores (x, y, z, w)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    out = np.eye(4)
    rs = _quat_matrix(node.get("rotation", [0, 0, 0, 1])) @ np.diag(node.get("scale", [1, 1, 1]))
    out[:3, :3] = rs
    out[:3, 3] = node.get("translation", [0, 0, 0])
    return out


def read_skeleton(path: Path) -> VrmSkeleton:
    doc = read_gltf_json(path)
    extensions = doc.get("extensions", {})
    if "VRMC_vrm" in extensions:
        vrm, version = extensions["VRMC_vrm"], "1.0"
        bones = {name: bone["node"] for name, bone in vrm["humanoid"]["humanBones"].items()}
        meta_keys = ("name", "authors", "licenseUrl", "avatarPermission", "commercialUsage",
                     "creditNotation", "modification")
    elif "VRM" in extensions:
        vrm, version = extensions["VRM"], "0.x"
        bones = {bone["bone"]: bone["node"] for bone in vrm["humanoid"]["humanBones"]
                 if bone.get("node", -1) >= 0}
        meta_keys = ("title", "author", "allowedUserName", "licenseName", "otherLicenseUrl",
                     "commercialUssageName", "violentUssageName", "sexualUssageName")
    else:
        raise ValueError(f"{path} is not a VRM (no VRMC_vrm or VRM extension)")
    nodes = doc["nodes"]

    parent_of: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = index
    world: dict[int, np.ndarray] = {}

    def world_matrix(index: int) -> np.ndarray:
        if index not in world:
            local = _local_matrix(nodes[index])
            parent = parent_of.get(index)
            world[index] = local if parent is None else world_matrix(parent) @ local
        return world[index]

    node_to_bone = {node: name for name, node in bones.items()}

    def humanoid_parent(index: int) -> str | None:
        current = parent_of.get(index)
        while current is not None:
            if current in node_to_bone:
                return node_to_bone[current]
            current = parent_of.get(current)
        return None

    positions = {name: world_matrix(node)[:3, 3].copy() for name, node in bones.items()}
    # Facing: toes in front of the feet. VRM 0.x models face −Z; turn them to face +Z.
    if "leftToes" in positions:
        turned = bool(positions["leftToes"][2] < positions["leftFoot"][2])
    else:
        turned = version == "0.x"
    if turned:
        positions = {name: p * np.array([-1.0, 1.0, -1.0]) for name, p in positions.items()}
    return VrmSkeleton(
        bone_nodes=bones,
        world_positions=positions,
        parents={name: humanoid_parent(node) for name, node in bones.items()},
        meta={key: vrm.get("meta", {}).get(key) for key in meta_keys},
        vrm_version=version,
        turned=turned,
    )
