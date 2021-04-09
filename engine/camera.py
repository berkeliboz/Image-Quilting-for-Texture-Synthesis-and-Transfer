import pygame.surfarray
import engine.entity as Entity
import pygame.surface
import pygame.math as math
import pygame.transform as transform
import config

class Camera(Entity.Transform):
    def __init__(self, size : math.Vector2):
        super().__init__()
        self.size = size
        self.camera_size = size.elementwise() * self.scale

    def render_entity(self, entity : Entity.Entity):
        if self.display:
            if entity.texture is None:
                return
            camera_location = entity.world_transform.location + self.location

            local_img = transform.scale(
                pygame.surfarray.make_surface(entity.texture),
                (int(entity.size[0] * entity.world_transform.scale[0]),
                 int(entity.size[1] * entity.world_transform.scale[1])))
            #TODO: Add clipping later
            self.display.blit(local_img, camera_location)


    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN]:
            self.location.y -= config.CAMERA_SPEED
        if keys[pygame.K_RIGHT]:
            self.location.x -= config.CAMERA_SPEED
        if keys[pygame.K_LEFT]:
            self.location.x += config.CAMERA_SPEED
        if keys[pygame.K_UP]:
            self.location.y += config.CAMERA_SPEED

        self.display.fill((0,0,0))
        self.top_left = math.Vector2(self.location[0] - self.camera_size[0]/2, self.location[1] - self.camera_size[1]/2)
        self.bot_right = self.top_left + self.camera_size

    def attach_display(self, display : pygame.Surface):
        self.display = display
