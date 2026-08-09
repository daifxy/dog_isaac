from isaaclab.utils import configclass
from dataclasses import MISSING

import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, mesh_terrains_cfg
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane, trimesh

import numpy as np
import isaaclab.sim as sim_utils
import os

from .terrain_func import my_gap_terrain

@configclass
class MyMeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    function = my_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    gap_spacing: float = MISSING
    platform_width: float = 1.0
