import pygame

import config
import engine.camera as camera

from pygame.math import Vector2
from engine.entity_manager import EntityManager
from engine.terrain import Quilting_thread
import threading
import time
FramePerSec = pygame.time.Clock()

class Display():
    def __init__(self):
        self.is_ticking = True
        pygame.init()
        self.display = pygame.display.set_mode((config.HEIGHT,config.WIDTH))

        pygame.display.set_caption("Title")
        self.display_camera = camera.Camera(Vector2(config.HEIGHT,config.WIDTH))
        self.display_camera.attach_display(self.display)

        self.patches = EntityManager.get_terrain().fix_terrain(5, 5, 0, 0)

        # self.quilting_thread = Quilting_thread(EntityManager.get_terrain(), 0,0,10,15)
        # self.quilting_thread.start()
        # Absolutely last function to call
        self.update()


    def start_display(self):
        self.is_ticking = True

    def end_display(self):
        self.is_ticking = False

    def update(self):
        while self.is_ticking:
            self.display_camera.update()
            # Add this code thread functionality
            if EntityManager.get_terrain() is not None:
                row, col = self.display_camera.calculate_render_target_root()

                # new_patches = self.quilting_thread.pop_buffer()
                # if new_patches:
                #     self.patches.extend(new_patches)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_ticking = False
                    return
            for entity in EntityManager.entity_heap:
                entity[1].update()
                self.display_camera.render_entity(entity[1])
            if len(self.patches) > 0:
                for patch in self.patches:
                    self.display_camera.render_entity(patch)

            pygame.display.flip()
