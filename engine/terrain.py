import config
import image_quilting
import numpy as np
import cv2 as cv

from engine.entity import Entity
from engine.transform import Transform
from direction import direction
from engine.entity_manager import EntityManager

class Terrain(Entity):
    def __init__(self, size, world_transform):
        super().__init__(size, world_transform)
        output_path = config.OUTPUT_PATH
        self.terrain_id_map = dict()

        # TODO: Deserialization
        #if path:
        # else:

        # Create Terrain
        self.request_terrain()
        img = image_quilting.read_path_2RGB(config.SAMPLE_IMAGE_PATH)
        self.random_patches = image_quilting.generate_path_list(img, 5000, config.TERRAIN_SAMPLE_PATCH_SIZE)

        self.request_terrain_extention(5, 4)
        self.request_terrain_extention(4, 5)
        self.request_terrain_extention(5, 3)
        self.request_terrain_extention(5, 2)
        self.request_terrain_extention(5, 1)
        self.request_terrain_extention(5, 0)
#        self.request_terrain_extention(5, 2)
#        self.request_terrain_extention(6, 3)


        #self.request_terrain_extention(5, 5)
        #self.request_terrain_extention(5, 7)

        #self.request_terrain_extention(6, 4)
        #self.request_terrain_extention(7, 4)

    def request_terrain(self):
        max_size = config.TERRAIN_MAX_NUMBER_OF_PATCHES
        sample_patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE
        texture = image_quilting.generate_texture(config.SAMPLE_IMAGE_PATH, sample_patch_size)

        rows = texture.shape[0]
        cols = texture.shape[1]

        number_of_horizontal_patches = int(cols / sample_patch_size)
        number_of_vertical_patches = int(rows / sample_patch_size)

        for row in range(number_of_vertical_patches - 1):
            for col in range(number_of_horizontal_patches - 1):
                patch = Terrain_Patch((sample_patch_size,sample_patch_size), Transform((row * sample_patch_size, col * sample_patch_size)),row,col, self)
                self.terrain_id_map[(row, col)] = patch
                patch.set_texture(texture[
                           row * sample_patch_size : ((row + 1) * sample_patch_size),
                           col * sample_patch_size : ((col + 1) * sample_patch_size)])

    def request_terrain_extention(self, row, col):
        quilt_size = config.TERRAIN_QUILTING_SIZE
        patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE
        dir = direction()

        horizontal_source_patch = None
        vertical_source_patch = None
        dir.Right = self.terrain_id_map.get((row, col - 1)) is not None
        dir.Up = self.terrain_id_map.get((row - 1, col)) is not None

        if dir.Right:
            # Grab patch on right as sample
            horizontal_source_patch = self.terrain_id_map[(row,col-1)].texture
        if dir.Up:
            # Grab patch below as sample
            vertical_source_patch = self.terrain_id_map[(row - 1,col)].texture

        result_patch = image_quilting.find_ssd(
            horizontal_source_patch,
            vertical_source_patch,
            self.random_patches, dir)

        #Extend Bottom
        if dir.Up:
            sample_dir = direction()
            sample_dir.Up = True
            res = image_quilting.stitch_vertical(vertical_source_patch, result_patch, self.random_patches)

            self.terrain_id_map[(row - 1, col)].set_texture(res[0:patch_size,:,:])
            result_patch = res[patch_size: patch_size + patch_size,:,:]

        # Extend Left
        if dir.Right:
            sample_dir = direction()
            sample_dir.Right = True
            res = image_quilting.stitch_horizontal(horizontal_source_patch, result_patch, self.random_patches)
            self.terrain_id_map[(row, col - 1)].set_texture(res[:, 0:patch_size, :])
            result_patch = res[:, patch_size: patch_size + patch_size, :]

        #TODO: Final Transform is wrong, recalculate it
        patch = Terrain_Patch((patch_size, patch_size),Transform((row * patch_size, col * patch_size)), row, col, self)
        self.terrain_id_map[(row,col)] = patch
        patch.set_texture(result_patch)

class Terrain_Patch(Entity):
    def __init__(self, size, world_transform, row,col, attached_terrain : Terrain):
        super().__init__(size, world_transform)
        attached_terrain.terrain_id_map[(row, col)] = self

    # def update_texture(self):
    #     EntityManager.entity_heap.__getitem__()