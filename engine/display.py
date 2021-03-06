import pygame

import config
import engine.camera as camera

from pygame.math import Vector2
from engine.entity_manager import EntityManager
from engine.terrain import Quilting_thread

FramePerSec = pygame.time.Clock()

class Display():
    def __init__(self):
        self.is_ticking = True
        pygame.init()
        self.display = pygame.display.set_mode((config.HEIGHT,config.WIDTH))
        self.row, self.col = 0, 0
        pygame.display.set_caption("Title")
        self.display_camera = camera.Camera(Vector2(config.HEIGHT,config.WIDTH))
        self.display_camera.attach_display(self.display)

        self.patches = []
        self.quilted_patches = {}

        self.row, self.col = self.display_camera.calculate_render_target_root()
        self.quilting_thread = Quilting_thread(EntityManager.get_terrain(), 1,1,self.display_camera.row_size, self.display_camera.col_size)
        self.quilting_thread.start()
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
                # self.quilting_thread.update_current_draw_start(row, col)
                new_patches = self.quilting_thread.pop_buffer()
                if new_patches:
                    self.patches.extend(new_patches)
                self.update_current_draw_start(row, col)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_ticking = False
                    return
            if len(self.patches) > 0:
                for patch in self.patches:
                    self.display_camera.render_terrain(patch)
            for entity in EntityManager.entity_heap:
                entity[1].update()
                self.display_camera.render_entity(entity[1])
            pygame.display.flip()

    def update_current_draw_start(self,row,col):
        redraw = False
        if (abs(self.col - col) > config.CAMERA_REDRAW_DISTANCE):
            self.col = col
            redraw = True
        if (abs(self.row - row) > config.CAMERA_REDRAW_DISTANCE):
            self.row = row
            redraw = True

        if redraw:
            # print("redrawing")
            self.quilted_patches = self.quilting_thread.quilted_patches
            new_patches = self.quilting_thread.pop_buffer()
            if new_patches:
                self.patches.extend(new_patches)
            self.quilting_thread.kill = True
            self.quilting_thread.join(0)
            self.quilting_thread = Quilting_thread(EntityManager.get_terrain(), self.row, self.col, self.display_camera.row_size, self.display_camera.col_size)
            self.quilting_thread.quilted_patches = self.quilted_patches
            self.quilting_thread.start()
