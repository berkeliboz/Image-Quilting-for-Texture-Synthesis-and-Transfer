import config
import image_quilting
import numpy as np
import cv2 as cv

from engine.entity import Entity
from engine.transform import Transform
from direction import direction

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
        self.request_terrain_extention(3, 5)
        self.request_terrain_extention(3, 6)
        self.request_terrain_extention(5, 3)
        self.request_terrain_extention(6, 3)
        self.request_terrain_extention(7, 3)
        self.request_terrain_extention(5, 4)
        self.request_terrain_extention(6, 4)
        self.request_terrain_extention(7, 4)

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
                self.terrain_id_map[(row, col)] = texture[
                          row * sample_patch_size : ((row + 1) * sample_patch_size),
                          col * sample_patch_size : ((col + 1) * sample_patch_size)]
                patch.set_texture(self.terrain_id_map[(row, col)])

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
            horizontal_source_patch = self.terrain_id_map[(row,col-1)]
        if dir.Up:
            # Grab patch below as sample
            vertical_source_patch = self.terrain_id_map[(row - 1,col)]

        result_patch = image_quilting.find_ssd(
            horizontal_source_patch,
            vertical_source_patch,
            self.random_patches, dir)
        #Extend Top
        if dir.Up:
            sample_dir = direction()
            sample_dir.Up = True
            bottom_sample = image_quilting.find_ssd(
                horizontal_source_patch,
                result_patch,
                self.random_patches, sample_dir)
            img = np.concatenate((vertical_source_patch,result_patch, bottom_sample))
            result = image_quilting.do_vertical_cut_and_stich(img,2, False)
            # TODO: This doesn't modify the image,
            # Solution: Delete the entity and recreate it.
            self.terrain_id_map[(row - 1, col)] = result[0:patch_size,:,:]
            result_patch = result[patch_size: patch_size + patch_size,:,:]

        #Extend Left

        # TODO: WIP
        # Extend Right
        if dir.Right:
            combined = image_quilting.combine_patch_horizontal(horizontal_source_patch, result_patch)
            sample_dir = direction()
            sample_dir.Right = True
            right_sample = image_quilting.find_ssd(
                result_patch,
                None,
                self.random_patches, sample_dir)
            combined_r = image_quilting.combine_patch_horizontal(result_patch, right_sample)
            last = np.concatenate((horizontal_source_patch[:, 0:patch_size - quilt_size, :],
                                   combined,
                                   result_patch[:, quilt_size:patch_size - quilt_size, :],
                                   combined_r,
                                   right_sample[:, quilt_size : quilt_size + quilt_size, :]),
                                  axis=1)
            self.terrain_id_map[(row,col-1)] = last[:,0: patch_size,:]
            result_patch = last[:, patch_size : patch_size + patch_size, :]

        patch = Terrain_Patch((patch_size, patch_size),Transform((row * patch_size, col * patch_size)), row, col, self)
        self.terrain_id_map[(row,col)] = result_patch
        patch.set_texture(self.terrain_id_map[(row,col)])

        #Extend Bottom

class Terrain_Patch(Entity):
    def __init__(self, size, world_transform, row,col, attached_terrain : Terrain):
        super().__init__(size, world_transform)
        attached_terrain.terrain_id_map[(row, col)] = self
