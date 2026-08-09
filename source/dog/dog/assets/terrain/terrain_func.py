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


@configclass
class MyMeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    gap_width_range: tuple[float, float] = MISSING
    gap_spacing: float = MISSING
    platform_width: float = 1.0

def my_gap_terrain(
    difficulty: float, cfg: MyMeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by gaps.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    a_gap = cfg.gap_spacing + gap_width
    min_width = min(cfg.size[0], cfg.size[1])
    num_gaps = int(((min_width - cfg.platform_width) / a_gap)/2)
    
    terrain_height = 1.
    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)

    # Generate the outer ring
    size = cfg.size
    for num_gap in range(num_gaps):
        inner_size = (size[0] - cfg.gap_spacing * 2, size[1] - cfg.gap_spacing * 2)
        meshes_list += make_border(size, inner_size, terrain_height, terrain_center)
        size = (size[0] - a_gap * 2, size[1] - a_gap * 2)


    # Generate the inner box
    box_dim = ((min_width - (num_gaps * a_gap * 2)), (min_width - (num_gaps * a_gap * 2)), terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)

    ground_dim = (cfg.size[0], cfg.size[1], 0.1)
    center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -1.)
    ground = trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(center))
    meshes_list.append(ground)


    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin

