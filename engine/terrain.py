import config
import image_quilting
import numpy as np
import cv2 as cv

from engine.entity import Entity
from engine.transform import Transform
from direction import direction

class Terrain_Patch(Entity):
    def __init__(self, size, world_transform):
        super().__init__(size, world_transform)


class Terrain(Entity):
    def __init__(self, size, world_transform):
        super().__init__(size, world_transform)
        output_path = config.OUTPUT_PATH

        # TODO: Deserialization
        #if path:
        # else:

        # Create Terrain
        self.patches, self.max_row, self.max_col = self.request_terrain()
        img = image_quilting.read_path_2RGB(config.SAMPLE_IMAGE_PATH)
        self.random_patches = image_quilting.generate_path_list(img, 5000, config.TERRAIN_SAMPLE_PATCH_SIZE)
        #self.request_terrain_extention(0, 5)

    def request_terrain(self):
        max_size = config.TERRAIN_MAX_NUMBER_OF_PATCHES
        sample_patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE
        texture = image_quilting.generate_texture(config.SAMPLE_IMAGE_PATH, sample_patch_size)

        rows = texture.shape[0]
        cols = texture.shape[1]

        number_of_horizontal_patches = int(cols / sample_patch_size)
        number_of_vertical_patches = int(rows / sample_patch_size)

        patches = np.empty((max_size, max_size), dtype= np.ndarray)
        for row in range(number_of_vertical_patches - 1):
            for col in range(number_of_horizontal_patches - 1):
                patch = Terrain_Patch((sample_patch_size,sample_patch_size), Transform((row * sample_patch_size, col * sample_patch_size)))
                patches[row][col] = texture[
                          row * sample_patch_size : ((row + 1) * sample_patch_size),
                          col * sample_patch_size : ((col + 1) * sample_patch_size)]
                patch.set_texture(patches[row][col])
        return patches, number_of_vertical_patches - 1, number_of_horizontal_patches - 1

    # TODO: 1 - Direction is redundant, can be calculated
    def request_terrain_extention(self, row, col):
        quilt_size = config.TERRAIN_QUILTING_SIZE
        patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE
        dir = direction()

        horizontal_source_patch = None
        vertical_source_patch = None
        dir.Right = self.patches[row][col - 1] is not None

        if dir.Right:
            horizontal_source_patch = self.patches[row][col - 1]
        #Extend Top
        #Extend Right

        #TODO: WIP
        #Extend Left
        if dir.Right:
            result_patch = image_quilting.find_ssd(
                horizontal_source_patch,
                vertical_source_patch,
                self.random_patches, dir)

            combined = image_quilting.combine_patch_horizontal(self.patches[row][col - 1], result_patch, patch_size)
            self.patches[row][col - 1] = combined
            result_patch = result_patch[:, quilt_size : patch_size, :]

            patch = Terrain_Patch((patch_size, patch_size),Transform((row * patch_size, col * patch_size)))
            self.patches[row][col] = result_patch
            patch.set_texture(self.patches[row][col])

            cv.waitKey(0)
        #Extend Bottom


