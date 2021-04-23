from engine.transform import Transform
from engine.entity import Entity

class Actor(Entity):
    def __init__(self, size, world_transform: Transform, enable_update):
        super().__init__(size, world_transform, enable_update)