from engine.entity_manager import EntityManager
from engine.transform import Transform

class Entity():
    def __init__(self, size, world_transform, enable_update):
        self.world_transform = world_transform
        self.local_transform = Transform()
        self.id = EntityManager.register_entity(self)
        self.size = size
        self.texture = None
        self.enable_update = enable_update
        if enable_update:
            EntityManager.enable_update(self,self.id)
    def __del__(self):
        EntityManager.free_id(self.id)

    def set_texture(self, texture):
        self.texture = texture

    def update(self):
        pass