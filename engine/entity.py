import pygame.math as math
import engine.entity_manager as entity_manager

class Transform():
    def __init__(self):
        self.location = math.Vector2(0, 100)
        self.rotation = math.Vector2(0, 0)
        self.scale = math.Vector2(1, 1)

class Entity():
    def __init__(self,size):
        self.world_transform = Transform()
        self.local_transform = Transform()
        self.id = entity_manager.EntityManager.register_entity(self)
        self.size = size
    def __del__(self):
        entity_manager.EntityManager.free_id(self.id)

    def set_texture(self, texture):
        self.texture = texture

    def update(self):
        pass
