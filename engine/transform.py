import pygame.math as math


class Transform:
    def __init__(self, location = None, rotation = None, scale = None):
        self.location = math.Vector2(0, 0) if location == None else location
        self.rotation = math.Vector2(0, 0) if rotation == None else rotation
        self.scale = math.Vector2(1, 1) if scale == None else scale

    def get_location_string(self):
        return str.format("-wt-{0}-{1}", self.location[0], self.location[1])
