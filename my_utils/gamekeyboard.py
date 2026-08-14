# from pynput.keyboard import Listener, Key


# class GameKeyboard:
#     """ Class to handle keyboard input for game control"""

#     def __init__(self):
#         self.cmd_1, self.cmd_2, self.cmd_3 = 0., 0., 0.
#         self.reset = False
#         self.esc = False
#         self.terrain_level = None
#         self.terrain_type = None

#         listener = Listener(on_press=self.cmd)
#         listener.start()

#     def cmd(self, direction):
#         if direction == Key.page_up:
#             self.terrain_type = 1
#         elif direction == Key.page_down:
#             self.terrain_type = -1
#         elif direction == Key.shift_l:
#             self.terrain_level = -1
#             self.cmd_1, self.cmd_2, self.cmd_3 = 0.0, 0.0, 0.0
#         elif direction == Key.space:
#             self.terrain_level = 1
#             self.cmd_1, self.cmd_2, self.cmd_3 = 0.0, 0.0, 0.0
#         elif direction == Key.esc:
#             self.esc = True
#         # Character key alternatives: w/s for vx, a/d for vy, q/e for yaw, r to reset
#         elif hasattr(direction, 'char'):
#             if direction.char == 'w':
#                 self.cmd_1 += 0.1
#             elif direction.char == 's':
#                 self.cmd_1 -= 0.1
#             elif direction.char == 'a':
#                 self.cmd_2 += 0.1
#             elif direction.char == 'd':
#                 self.cmd_2 -= 0.1
#             elif direction.char == 'q':
#                 self.cmd_3 += 0.1
#             elif direction.char == 'e':
#                 self.cmd_3 -= 0.1
#             elif direction.char == 'r':
#                 self.reset = True
#         print(f"cmd: vx={self.cmd_1:.2f}, vy={self.cmd_2:.2f}, yaw={self.cmd_3:.2f}")

#     def get_commands(self):
#         reset = self.reset
#         esc = self.esc
#         terrain_level = self.terrain_level
#         terrain_type = self.terrain_type

#         self.reset = False
#         self.esc = False
#         self.terrain_level = None
#         self.terrain_type = None

#         return [self.cmd_1, self.cmd_2, self.cmd_3], reset, esc, terrain_level, terrain_type

# class Command:
#     vx, vy, dyaw = 0.0, 0.0, 0.0
#     vx_increment = 0.1
#     vy_increment = 0.1
#     dyaw_increment = 0.1

#     min_vx = -1.0
#     max_vx = 2.0
#     min_vy = -1.0
#     max_vy = 1.0
#     min_dyaw = -1.0
#     max_dyaw = 1.0
#     reset_requested = False

#     def __init__(self):
#         self.reset()
#         listener = keyboard.Listener(on_press=self.on_press)
#         listener.start()

#     def reset(self):
#         """reset all velocities to zero"""
#         self.vx = 0.0
#         self.vy = 0.0
#         self.dyaw = 0.0
#         self.reset_requested = True
#         self.terminate = False
#         print(f"Velocities reseted. ")

#     def advance(self):
#         """get current command"""
#         vx, vy, dyaw = self.vx, self.vy, self.dyaw
#         is_reset = self.reset_requested
#         terminate = self.terminate
#         self.reset_requested = False
#         return [vx, vy, dyaw], is_reset, terminate

#     def on_press(self, key):
#         """Key press event handler"""
#         try:
#             # Number key controls: 8/5 control forward/backward (vx), 4/6 control left/right (vy), 7/9 control left/right turn (dyaw)
#             if hasattr(key, 'char') and key.char is not None:
#                 c = key.char.lower()
#                 if c == '8':
#                     self.vx = np.clip(self.vx + Command.vx_increment, self.min_vx, self.max_vx)
#                 elif c == '2':
#                     self.vx = np.clip(self.vx - Command.vx_increment, self.min_vx, self.max_vx)
#                 elif c == '4':
#                     self.vy = np.clip(self.vy + Command.vy_increment, self.min_vy, self.max_vy)
#                 elif c == '6':
#                     self.vy = np.clip(self.vy - Command.vy_increment, self.min_vy, self.max_vy)
#                 elif c == '7':
#                     self.dyaw = np.clip(self.dyaw + Command.dyaw_increment, self.min_dyaw, self.max_dyaw)
#                 elif c == '9':
#                     self.dyaw = np.clip(self.dyaw - Command.dyaw_increment, self.min_dyaw, self.max_dyaw)
#                 elif c == '5':
#                     self.terminate = True
#                 elif c == '0':
#                     self.reset()
#             print(f"cmd: {self.vx:.2f}, {self.vy:.2f}, {self.dyaw:.2f}")
#         except AttributeError:
#             pass
