from pynput.keyboard import Listener, Key


class GameKeyboard:
    """ Class to handle keyboard input for game control"""

    def __init__(self):
        self.cmd_1, self.cmd_2, self.cmd_3 = 0., 0., 0.
        self.reset = False
        self.esc = False
        self.terrain_level = None
        self.terrain_type = None

        listener = Listener(on_press=self.cmd)
        listener.start()

    def cmd(self, direction):
        if direction == Key.page_up:
            self.terrain_type = 1
        elif direction == Key.page_down:
            self.terrain_type = -1
        elif direction == Key.shift_l:
            self.terrain_level = -1
            self.cmd_1, self.cmd_2, self.cmd_3 = 0.0, 0.0, 0.0
        elif direction == Key.space:
            self.terrain_level = 1
            self.cmd_1, self.cmd_2, self.cmd_3 = 0.0, 0.0, 0.0
        elif direction == Key.esc:
            self.esc = True
        # Character key alternatives: w/s for vx, a/d for vy, q/e for yaw, r to reset
        elif hasattr(direction, 'char'):
            if direction.char == 'w':
                self.cmd_1 += 0.1
            elif direction.char == 's':
                self.cmd_1 -= 0.1
            elif direction.char == 'a':
                self.cmd_2 += 0.1
            elif direction.char == 'd':
                self.cmd_2 -= 0.1
            elif direction.char == 'q':
                self.cmd_3 += 0.1
            elif direction.char == 'e':
                self.cmd_3 -= 0.1
            elif direction.char == 'r':
                self.reset = True
        print(f"cmd: vx={self.cmd_1:.2f}, vy={self.cmd_2:.2f}, yaw={self.cmd_3:.2f}")

    def get_commands(self):
        reset = self.reset
        esc = self.esc
        terrain_level = self.terrain_level
        terrain_type = self.terrain_type

        self.reset = False
        self.esc = False
        self.terrain_level = None
        self.terrain_type = None

        return [self.cmd_1, self.cmd_2, self.cmd_3], reset, esc, terrain_level, terrain_type
