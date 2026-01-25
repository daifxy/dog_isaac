from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
import numpy as np
import torch

class _MyPrint_:
    Format = {
    # 基础颜色
    "BLACK" : '\033[30m',
    "RED" : '\033[31m',
    "GREEN" : '\033[32m',
    "YELLOW" : '\033[33m',
    "BLUE" : '\033[34m',
    "MAGENTA" : '\033[35m',
    "CYAN" : '\033[36m',
    "WHITE" : '\033[37m',
    
    # 亮色
    "BRIGHT_BLACK" : '\033[90m',
    "BRIGHT_RED" : '\033[91m',
    "BRIGHT_GREEN" : '\033[92m',
    "BRIGHT_YELLOW" : '\033[93m',
    "BRIGHT_BLUE" : '\033[94m',
    "BRIGHT_MAGENTA" : '\033[95m',
    "BRIGHT_CYAN" : '\033[96m',
    "BRIGHT_WHITE" : '\033[97m',
    
    # 背景色
    "BG_BLACK" : '\033[40m',
    "BG_RED" : '\033[41m',
    "BG_GREEN" : '\033[42m',
    "BG_YELLOW" : '\033[43m',
    "BG_BLUE" : '\033[44m',
    "BG_MAGENTA" : '\033[45m',
    "BG_CYAN" : '\033[46m',
    "BG_WHITE" : '\033[47m',
    
    # 样式
    "BOLD" : '\033[1m',
    "DIM" : '\033[2m',
    "ITALIC" : '\033[3m',
    "UNDERLINE" : '\033[4m',
    "BLINK" : '\033[5m',
    "REVERSE" : '\033[7m',
    "HIDDEN" : '\033[8m',
    "UNDERLINE_ITALIC" : '\033[3m\033[4m',

    # 事件
    "ERROR" : '\033[91m[ERROR]: ',
    "SUCCESS" : '\033[92m',
    "WARNING" : '\033[93m[WARNING] ',
    "INFO" : '\033[94m[INFO]: ',
    "DEBUG" : '\033[95m',

    # 重置所有样式
    "RESET" : '\033[0m',
    }

    def __call__(self, msg: object = "", Type: str | None = None, Interrupt: bool = False):
        self.type = self.Format["WHITE"]
        if Type in self.Format.keys():
            self.type = self.Format[Type]
        self._msg = f"{self.type}{msg}{self.Format['RESET']}"
        print(self._msg)
        if Interrupt:raise Exception(msg)

    def __enter__(self):
        print(self.type[:5], end='')
        return self
    
    def __exit__(self, *args):
        if self.type == self.Format["ERROR"]:
            raise Exception()
        print(self.Format["RESET"], end='')


class Visualization:
    def __init__(self, num_envs: int, device):
        self.num_envs = num_envs
        self.all_envs = torch.arange(num_envs)
        self.visualize_markers: VisualizationMarkers = self.define_markers()
        self.visualize_markers.visualize(torch.zeros((num_envs, 3),device=device, dtype=torch.float32),
                                        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32).expand((num_envs, 4)),
                                        torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32).expand((num_envs, 3)),
                                        [1]
                                )

    def visualize(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        marker_indices: list[int] | np.ndarray | torch.Tensor | None = None,
    ):
        """Visualize markers at the given positions and orientations."""
        indices = [torch.full_like(self.all_envs, idx) for idx in marker_indices]
        _indices = torch.hstack(indices)
        self.visualize_markers.visualize(translations, orientations, scales, _indices)


    def define_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "arrow_red": sim_utils.UsdFileCfg( # 箭头
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "arrow_green": sim_utils.UsdFileCfg( # 箭头
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "arrow_blue": sim_utils.UsdFileCfg( # 箭头
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                "frame": sim_utils.UsdFileCfg( # 坐标系轴
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
                ),
                "cube": sim_utils.CuboidCfg( # 立方体
                    size=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                ),
                "sphere": sim_utils.SphereCfg( # 球体
                    radius=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "cylinder": sim_utils.CylinderCfg( # 圆柱体
                    radius=0.5,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                "cone": sim_utils.ConeCfg( # 圆锥体
                    radius=0.5,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
                # "mesh": sim_utils.UsdFileCfg( # 有字母的立方体
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                #     scale=(10.0, 10.0, 10.0),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.0)),
                # ),
            },
        )
        return VisualizationMarkers(cfg=marker_cfg)