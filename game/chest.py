import pygame
import config

from engine.transform import Transform
from game.actor import Actor


class Chest(Actor):
    def __init__(self, size, world_transform: Transform, enable_update):
        super().__init__(size, world_transform, enable_update)

        world_transform.scale.y *= 1
        self.texture = pygame.image.load(config.GAME_CHEST)
