import pygame.surfarray
import engine.entity as Entity
import pygame.surface
import pygame.math as math
import pygame.transform as transform
from engine.transform import Transform
from engine.entity_manager import EntityManager
from engine.terrain import Terrain
import cv2 as cv
import config


class Camera(Transform):
    def __init__(self, size: math.Vector2):
        super().__init__()
        self.size = size
        self.camera_size = size.elementwise() * self.scale
        self.render_target_start = (0, 0)
        self.render_size = 0
        self.row_size, self.col_size = self.calculate_grid_size()
        self.attached_terrain = None

    def render_entity(self, entity: Entity.Entity):
        if self.display:
            if entity.texture is None:
                return
            camera_location = entity.world_transform.location + self.location

            local_img = transform.scale(
                pygame.surfarray.make_surface(entity.texture),
                (int(entity.size[0] * entity.world_transform.scale[0]),
                 int(entity.size[1] * entity.world_transform.scale[1])))

            self.display.blit(local_img, camera_location)

    def calculate_grid_size(self):
        patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE - config.TERRAIN_QUILTING_SIZE
        rows = int(config.HEIGHT / patch_size)
        cols = int(config.WIDTH / patch_size)
        return rows, cols

    def calculate_render_target_root(self):
        x = -self.location.x
        y = -self.location.y
        patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE - config.TERRAIN_QUILTING_SIZE
        render_x = int(x / patch_size)
        render_y = int(y / patch_size)
        return render_x, render_y

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

        self.display.fill((0, 0, 0))
        self.top_left = math.Vector2(self.location[0] - self.camera_size[0] / 2,
                                     self.location[1] - self.camera_size[1] / 2)
        self.bot_right = self.top_left + self.camera_size

    def attach_display(self, display: pygame.Surface):
        self.display = display