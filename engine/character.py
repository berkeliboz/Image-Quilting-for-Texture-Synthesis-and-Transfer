import pygame
import config
from engine.entity import Entity

from engine.transform import Transform

class Character(Entity):
    def __init__(self, size, world_transform : Transform, enable_update):
        super().__init__(size, world_transform, enable_update)
        self.world_transform = world_transform
        self.world_transform.scale.x = int(size[0]/100)
        self.world_transform.scale.y = int(size[1]/100)

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.world_transform.location.y -= config.CAMERA_SPEED
        if keys[pygame.K_d]:
            self.world_transform.location.x += config.CAMERA_SPEED
        if keys[pygame.K_a]:
            self.world_transform.location.x -= config.CAMERA_SPEED
        if keys[pygame.K_s]:
            self.world_transform.location.y += config.CAMERA_SPEED