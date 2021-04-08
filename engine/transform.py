import pygame.math as math

class Transform():
    def __init__(self, location = None, rotation = None, scale = None):
        self.location = math.Vector2(0, 0) if location == None else location
        self.rotation = math.Vector2(0, 0) if rotation == None else rotation
        self.scale = math.Vector2(1, 1) if scale == None else scale