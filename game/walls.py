import pygame
import config

from engine.transform import Transform
from game.actor import Actor


class Walls:
    def __init__(self):
        for i in range(11):
            Wall((100, 100), (Transform(pygame.Vector2(16 + i * 96, -30))), True)


class Wall(Actor):
    def __init__(self, size, world_transform: Transform, enable_update):
        super().__init__(size, world_transform, enable_update)

        world_transform.scale.y *=1.5
        self.texture = pygame.image.load(config.GAME_WALL)