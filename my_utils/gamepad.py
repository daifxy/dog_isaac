import pygame
import numpy as np
from pathlib import Path

class control_gamepad:
    def __init__(self,command_cfg: dict):
        pygame.init()
        pygame.joystick.init()
        # 初始化控制窗口
        screen_width = 500
        screen_height = 500
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        image_path = "picture/keyboard_key.png"
        image_center = (250, 250)
        pygame.display.set_caption("请用此窗口进行键盘控制(This use your keyboard)")
        image_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
        image_surface.fill((255, 255, 255, 0))  # 透明初始化
        try:
            if Path(image_path).exists():
                image_surface = pygame.image.load(image_path)
                scaled_image = pygame.transform.smoothscale(image_surface, (500, 500))        
            else:
                print(f"无法加载图片: picture/keyboard_key.png")
        except pygame.error as e:
            print(f"无法加载图片: picture/keyboard_key.png")
            print(e)
            pygame.quit()
            exit()
        image_rect = scaled_image.get_rect()
        image_rect.center = image_center
        self.screen.fill((255, 255, 255)) # 背景
        self.screen.blit(scaled_image, image_rect)
        pygame.display.flip() # 更新屏幕显示 (一次性) 

        self.command_cfg = command_cfg
        self.commands = np.zeros(command_cfg["num_commands"])
        self.stand_flag: bool = False

    def get_commands(self):
        pygame.event.pump()
        reset_flag = False
        terrain_id = None
        terrain_level = None
        for event in pygame.event.get():  # 获取事件队列中的所有事件
            if event.type == pygame.QUIT:  # 用户点击窗口关闭按钮
                running = False
            elif event.type == pygame.KEYDOWN:  # 键盘按键按下事件
                match event.key:
                    case pygame.K_w:
                        self.commands[0] = self.command_cfg["lin_vel_x_range"][1] * 0.5 if self.stand_flag else self.command_cfg["lin_vel_x_range"][1]
                    case pygame.K_s:
                        self.commands[0] = self.command_cfg["lin_vel_x_range"][0] * 0.5 if self.stand_flag else self.command_cfg["lin_vel_x_range"][0]
                    case pygame.K_a:
                        self.commands[1] = self.command_cfg["lin_vel_y_range"][1] * 0.5 if self.stand_flag else self.command_cfg["lin_vel_y_range"][1]
                    case pygame.K_d:
                        self.commands[1] = self.command_cfg["lin_vel_y_range"][0] * 0.5 if self.stand_flag else self.command_cfg["lin_vel_y_range"][0] 
                    case pygame.K_q:
                        self.commands[2] = self.command_cfg["ang_vel_range"][0] * 0.5 if self.stand_flag else self.command_cfg["ang_vel_range"][1]
                    case pygame.K_e:
                        self.commands[2] = self.command_cfg["ang_vel_range"][1] * 0.5 if self.stand_flag else self.command_cfg["ang_vel_range"][0]
                    case pygame.K_LSHIFT:
                        terrain_id = 1
                    case pygame.K_SPACE:
                        terrain_id = -1
                    case pygame.K_PAGEUP:
                        terrain_level = 1
                    case pygame.K_PAGEDOWN:
                        terrain_level = -1
                    case pygame.K_r:
                        reset_flag=True
                        
            elif event.type == pygame.KEYUP:  # 键盘按键释放事件
                match event.key:
                    case pygame.K_w:
                        self.commands[0] = 0
                    case pygame.K_s:
                        self.commands[0] = 0
                    case pygame.K_a:
                        self.commands[1] = 0
                    case pygame.K_d:
                        self.commands[1] = 0
                    case pygame.K_q:
                        self.commands[2] = 0
                    case pygame.K_e:
                        self.commands[2] = 0
                        
        self.commands_clip()
        return self.commands, reset_flag, terrain_id, terrain_level
    
    def commands_clip(self):
        # lin_vel_x
        if (self.commands[0] <= self.command_cfg["lin_vel_x_range"][0]):
            self.commands[0] = self.command_cfg["lin_vel_x_range"][0]
        elif (self.commands[0] >= self.command_cfg["lin_vel_x_range"][1]):
            self.commands[0] = self.command_cfg["lin_vel_x_range"][1]

        # lin_vel_y
        if (self.commands[1] <= self.command_cfg["lin_vel_y_range"][0]):
            self.commands[1] = self.command_cfg["lin_vel_y_range"][0]
        elif (self.commands[1] >= self.command_cfg["lin_vel_y_range"][1]):
            self.commands[1] = self.command_cfg["lin_vel_y_range"][1]

        # ang_vel
        if (self.commands[2] <= self.command_cfg["ang_vel_range"][0]):
            self.commands[2] = self.command_cfg["ang_vel_range"][0]
        elif (self.commands[2] >= self.command_cfg["ang_vel_range"][1]):
            self.commands[2] = self.command_cfg["ang_vel_range"][1]



class GamepadSimple:
    """直接读取 Linux /dev/input/jsX 设备文件，输出 SE(2) 速度指令。

    """

    def __init__(self, vx=1.0, vy=1.0, wz=2.0, dead_zone=0.1, dev="/dev/input/js0"):
        import os
        self._vx = vx
        self._vy = vy
        self._wz = wz
        self._dead_zone = dead_zone
        try:
            self._fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
            self._axes = [0.0] * 8
            self._buttons = [0] * 16
            self._connected = True
            print(f"手柄已连接: {dev}")
        except FileNotFoundError:
            self._fd = None
            self._connected = False
            print(f"未检测到手柄 ({dev})，返回零指令")

    def _poll(self):
        if not self._connected:
            return
        import struct, os
        try:
            while True:
                data = os.read(self._fd, 8)
                if not data:
                    break
                _time, value, ev_type, number = struct.unpack('IhBB', data)
                ev_type &= 0x7f
                if ev_type == 2:
                    self._axes[number] = value / 32767.0
                elif ev_type == 1:
                    self._buttons[number] = value
        except BlockingIOError:
            pass

    def advance(self) -> np.ndarray:
        """每帧调用，返回当前摇杆对应的速度指令。"""
        self._poll()
        cmd = np.zeros(3, dtype=np.float32)
        ly = self._axes[1]    # 左摇杆 Y 轴 (向上为负)
        lx = self._axes[0]    # 左摇杆 X 轴
        rx = self._axes[3]    # 右摇杆 X 轴（用于转向）
        reset = False
        if abs(ly) > self._dead_zone:
            cmd[0] = -ly * self._vx    # 前进/后退（取反修正方向）
        if abs(lx) > self._dead_zone:
            cmd[1] = -lx * self._vy    # 左移/右移
        if abs(rx) > self._dead_zone:
            cmd[2] = -rx * self._wz    # 左转/右转
        if self._buttons[1] >= 1.0:
            reset = True
        death = cmd[:] < 0.1
        cmd[death] = 0.0
        return cmd, reset, 0