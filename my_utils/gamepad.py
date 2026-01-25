import pygame
import numpy as np
from pathlib import Path

class control_gamepad:
    def __init__(self,command_cfg: dict):
        pygame.init()
        pygame.joystick.init()
        # 初始化控制窗口
        screen_width = 400
        screen_height = 580
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        image_path = "picture/keyboard_key.png" #sim2sim
        image_center = (200, 280)
        pygame.display.set_caption("请用此窗口进行键盘控制(This use your keyboard)")
        image_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
        image_surface.fill((255, 255, 255, 0))  # 透明初始化
        try:
            if Path(image_path).exists():
                image_surface = pygame.image.load(image_path)
            else:
                print(f"无法加载图片: picture/keyboard_key.png")
        except pygame.error as e:
            print(f"无法加载图片: picture/keyboard_key.png")
            print(e)
            pygame.quit()
            exit()
        image_rect = image_surface.get_rect()
        image_rect.center = image_center
        self.screen.fill((255, 255, 255)) # 背景
        self.screen.blit(image_surface, image_rect)
        pygame.display.flip() # 更新屏幕显示 (一次性) 

        self.command_cfg = command_cfg
        self.commands = np.zeros(command_cfg["num_commands"])
        self.stand_flag: bool = False

    def get_commands(self):
        pygame.event.pump()
        reset_flag = False
     
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
                        safe_linv_x = np.clip(np.abs(self.commands[0]), a_min=1e-2, a_max=1000)
                        angv_limit = self.command_cfg["inverse_linx_angv"] / safe_linv_x
                        self.commands[2] = np.min([angv_limit, self.command_cfg["ang_vel_range"][1]])
                        if self.stand_flag:
                            self.commands[2] = np.clip(self.commands[2], a_min=0, a_max=self.command_cfg["ang_vel_range"][1]/2)
                    case pygame.K_e:
                        safe_linv_x = np.clip(np.abs(self.commands[0]), a_min=1e-2, a_max=1000)
                        angv_limit = self.command_cfg["inverse_linx_angv"] / safe_linv_x
                        self.commands[2] = np.max([-angv_limit, self.command_cfg["ang_vel_range"][0]])
                        if self.stand_flag:
                            self.commands[2] = np.clip(self.commands[2], a_min=self.command_cfg["ang_vel_range"][0]/2, a_max=0)
                    case pygame.K_0:
                        self.commands[3] = 0
                        self.stand_flag = False
                    case pygame.K_1:
                        self.commands[3] = 1
                        self.stand_flag = True
                    case pygame.K_2:
                        self.commands[3] = 2
                        self.stand_flag = True
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
        return self.commands,reset_flag
    
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