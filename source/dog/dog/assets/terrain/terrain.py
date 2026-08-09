"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
from isaaclab.sim import RigidBodyMaterialCfg
import isaaclab.sim as sim_utils
import os

from .terrain_class import *

# 或者一行代码
parent_dir = os.path.dirname(__file__)

USDPATH = parent_dir + "/Collected_complete_enue_mesh/complete_enue_mesh.usd"

# ROUGH_TERRAINS_GENERATOR_CFG = TerrainGeneratorCfg(
#     size=(10., 10.),
#     border_width=25.0,
#     num_rows=9,
#     num_cols=5,
#     curriculum=True,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=True,
#     sub_terrains={
#         # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#         #     proportion=0.25,
#         #     step_height_range=(0.0, 0.15),
#         #     step_width=0.3,
#         #     platform_width=1.5,
#         #     border_width=0.5,
#         #     holes=False,
#         # ),
#         "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.0, 0.16),
#             step_width=0.3,
#             platform_width=4,
#             border_width=0.3,
#             holes=False,
#         ),
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.15), platform_width=1.5
#         ),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2, noise_range=(0.01, 0.1), noise_step=0.02, border_width=0.25
#         ),
#         "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.2, slope_range=(0.0, 0.2), platform_width=1.5, border_width=0.25
#         ),
#         "plane": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.2, slope_range=(0.0, 0.0), platform_width=1.5, border_width=0.25
#         ),
#     },
# )
"""Rough terrains configuration."""
ROUGH_TERRAINS_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(8., 8.),
    border_width=3.0,
    num_rows=30,
    num_cols=3,
    curriculum=True,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        "inv_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.333,
            step_height_range=(0.0, 0.20),
            step_width=0.3,
            platform_width=1.5,
            border_width=0.5,
            holes=False,
        ),
        # "gap": MyMeshGapTerrainCfg(
        #     proportion=0.25,
        #     gap_width_range=(0.0, 0.15),
        #     gap_spacing=0.4,
        #     platform_width=1.5,        
        # ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.333, grid_width=0.15, grid_height_range=(0.0, 0.15), platform_width=1.5
        ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.25, slope_range=(0.0, 0.0), platform_width=1.5, border_width=0.25
        # ),
        "platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.333, pit_depth_range=(0.0, 0.8), platform_width=1.5, double_pit=True
        ),
    },
)


ROUGH_TERRAIN = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_GENERATOR_CFG,
    collision_group=-1,
    debug_vis=False,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        project_uvw=True,
    ),
)


PLANE = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.6,
        dynamic_friction=1.4,
        restitution=0.0,
    ),
    debug_vis=False,
)


COMPLETE_ENUE_MESH = AssetBaseCfg(
    prim_path="/World/terrain",
    init_state = AssetBaseCfg.InitialStateCfg(
        pos=(0., 0., 0.01),
        rot=(1., 0., 0., 0.)
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USDPATH,
        scale=[0.1, 0.1, 0.1],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=False,
        ),
    ),
    collision_group=-1,
    debug_vis=True,
)