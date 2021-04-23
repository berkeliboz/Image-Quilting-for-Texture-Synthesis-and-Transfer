import pygame
import config

from engine.transform import Transform
from engine.character import Character

class Hero(Character):
    def __init__(self, size, world_transform: Transform, enable_update):
        super().__init__(size, world_transform, enable_update)
        self.texture = pygame.image.load(config.CHARACTER_IDLE_PATH)