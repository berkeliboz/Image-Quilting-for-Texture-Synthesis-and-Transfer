import os

import config
import image_quilting
import numpy as np
import cv2 as cv
import pickle

from engine.entity import Entity
from engine.transform import Transform
from direction import direction
from engine.entity_manager import EntityManager


class Terrain(Entity):
    def __init__(self, size, world_transform, enable_update):
        super().__init__(size, world_transform, enable_update)
        output_path = config.OUTPUT_PATH
        self.terrain_id_map = dict()

        # TODO: Deserialization
        # Moved the serialization to the individual texture patches and generate_terrain
        # Can be changed, but I found it was the best location
        # if path:
        # else:

        self.patches_cache = dict()
        img = image_quilting.read_path_2RGB(config.SAMPLE_IMAGE_PATH)
        self.random_patches = image_quilting.generate_path_list(img, config.TERRAIN_SAMPLE_PATCH_SIZE)
        self.generate_terrain(20, 20)

    def get_basis_texture_at(self, row, col):
        bt = self.terrain_id_map.get((row, col))
        if bt is None:
            return None
        return bt.basis_texture

    def set_texture_at(self, row, col, texture):
        self.terrain_id_map[(row, col)].texture = texture

    # Use this for direct texture modifications
    def get_texture_at(self, row, col):
        bt = self.terrain_id_map.get((row, col))
        if bt is None:
            return None
        return bt.texture

    def get_patch(self, row, col):
        return self.terrain_id_map[(row, col)]

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
                patch = Terrain_Patch((sample_patch_size, sample_patch_size),
                                      Transform((row * sample_patch_size, col * sample_patch_size)), False)
                patch.set_texture(texture[
                                  row * sample_patch_size: ((row + 1) * sample_patch_size),
                                  col * sample_patch_size: ((col + 1) * sample_patch_size)])

    def generate_terrain(self, number_of_horizontal_patches, number_of_vertical_patches):
        quilt_size = config.TERRAIN_QUILTING_SIZE
        sample_patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE

        # Case 0,0
        previous_texture = self.random_patches[0]
        patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform((0, 0)), False)
        self.terrain_id_map[(0, 0)] = patch
        patch.set_basis_texture(previous_texture)

        for row in range(number_of_vertical_patches):
            for col in range(1, number_of_horizontal_patches):

                patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform(
                    (row * (sample_patch_size - quilt_size), col * (sample_patch_size - quilt_size))), False)
                self.terrain_id_map[(row, col)] = patch
                if patch.has_basis_texture():
                    # print("Used cached file " + patch.get_basis_filename())
                    continue

                if row == 0:
                    result_texture = image_quilting.find_ssd(self.get_basis_texture_at(row, col - 1), None,
                                                             self.random_patches, direction(False, True, False, False))
                else:
                    horizontal = self.get_basis_texture_at(row, col - 1)
                    vertical = self.get_basis_texture_at(row - 1, col)
                    result_texture = image_quilting.find_ssd(horizontal, vertical, self.random_patches,
                                                             direction(True, True, False, False))
                patch.set_basis_texture(result_texture)

            # Case 0 N, grab first texture of previous row
            result_texture = image_quilting.find_ssd(None, self.get_basis_texture_at(row, 0), self.random_patches,
                                                     direction(True, False, False, False))

            patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform((row * sample_patch_size, 0)),
                                  False)
            self.terrain_id_map[(row + 1, 0)] = patch
            patch.set_basis_texture(result_texture)

    def fix_terrain(self, number_of_horizontal_patches, number_of_vertical_patches, row_offset, col_offset):
        patches = []
        for row in range(row_offset, number_of_vertical_patches + row_offset):
            for col in range(col_offset, number_of_horizontal_patches + col_offset - 1):
                texture = image_quilting.stitch_horizontal_lossy(self.get_basis_texture_at(row, col),
                                                                 self.get_basis_texture_at(row, col + 1))
                new_size = int(texture.shape[1] / 2)
                new_size_middle = int(new_size / 2)

                self.get_texture_at(row, col)[:, new_size_middle: new_size] = texture[:,
                                                                              new_size_middle: new_size_middle + new_size_middle]
                self.get_texture_at(row, col + 1)[:, 0: new_size_middle] = texture[:,
                                                                           new_size: new_size + new_size_middle]

        for row in range(number_of_vertical_patches - 1):
            for col in range(number_of_horizontal_patches):
                image = image_quilting.stitch_vertical_lossy(self.get_texture_at(row, col),
                                                             self.get_texture_at(row + 1, col))
                new_size = int(image.shape[0] / 2)
                new_size_middle = int(new_size / 2)

                self.get_texture_at(row, col)[new_size_middle: new_size, :] = image[
                                                                              new_size_middle: new_size_middle + new_size_middle,
                                                                              :]
                self.get_texture_at(row + 1, col)[0: new_size_middle, :] = image[new_size: new_size + new_size_middle,
                                                                           :]

                patches.append(self.get_patch(row, col))
        return patches

    def request_terrain_extention(self, row, col):
        quilt_size = config.TERRAIN_QUILTING_SIZE
        patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE
        dir = direction()

        horizontal_source_patch = None
        vertical_source_patch = None
        dir.Right = self.terrain_id_map.get((row, col - 1)) is not None
        # dir.Left = self.terrain_id_map.get((row, col + 1)) is not None
        dir.Up = self.terrain_id_map.get((row - 1, col)) is not None
        # dir.Down = self.terrain_id_map.get((row + 1, col)) is not None

        if dir.Right:
            # Grab patch on right as sample
            horizontal_source_patch = self.terrain_id_map[(row, col - 1)].texture
        if dir.Up:
            # Grab patch below as sample
            vertical_source_patch = self.terrain_id_map[(row - 1, col)].texture
            cv.imshow("vsp", vertical_source_patch)
        if dir.Down:
            # Grab patch below as sample
            vertical_source_patch = self.terrain_id_map[(row + 1, col)].texture
        if dir.Left:
            # Grab patch below as sample
            horizontal_source_patch = self.terrain_id_map[(row, col + 1)].texture
            cv.imshow("hsp", horizontal_source_patch)

        result_patch = image_quilting.find_ssd(
            horizontal_source_patch,
            vertical_source_patch,
            self.random_patches, dir)
        cv.imshow("res", result_patch)
        cv.waitKey(0)

        # Extend Bottom
        if dir.Up:
            sample_dir = direction()
            sample_dir.Up = True
            res = image_quilting.stitch_vertical(vertical_source_patch, result_patch, self.random_patches)

            # #Fix left
            self.terrain_id_map[(row - 1, col)].set_texture(res[0:patch_size, :, :])
            result_patch = res[patch_size: patch_size + patch_size, :, :]

        if dir.Down:
            sample_dir = direction()
            sample_dir.Down = True
            res = image_quilting.stitch_vertical(result_patch, vertical_source_patch, self.random_patches)

            # Fix left
            self.terrain_id_map[(row + 1, col)].set_texture(res[patch_size:patch_size + patch_size, :, :])
            result_patch = res[0: patch_size, :, :]

        # if dir.Left:
        #     sample_dir = direction()
        #     sample_dir.Left = True
        #     res = image_quilting.stitch_horizontal(result_patch,horizontal_source_patch, self.random_patches)
        #
        #     #Fix Up
        #     self.terrain_id_map[(row, col + 1)].set_texture(res[:, patch_size:patch_size + patch_size, :])
        #     result_patch = res[:, 0: patch_size, :]

        # Extend Left
        if dir.Right:
            sample_dir = direction()
            sample_dir.Right = True
            res = image_quilting.stitch_horizontal(horizontal_source_patch, result_patch, self.random_patches)
            # Fix Up
            # self.terrain_id_map[(row, col - 1)].set_texture(res[:, 0:patch_size, :])
            # result_patch = res[:, patch_size: patch_size + patch_size, :]
        # print(dir)
        # TODO: Final Transform is wrong, recalculate it
        patch = Terrain_Patch((patch_size, patch_size), Transform((row * patch_size, col * patch_size)), False)
        self.terrain_id_map[(row, col)] = patch
        patch.set_texture(result_patch)


class Terrain_Patch(Entity):
    def __init__(self, size, world_transform, enable_update):
        super().__init__(size, world_transform, False)

        # Check if basis image exists in a pickle, if so, load that
        self.pickle_basis_location = str.format("./engine/patches/{0}-{1}-basis.p", config.IMAGE_NAME,
                                                world_transform.get_location_string())
        if os.path.isfile(self.pickle_basis_location):
            self.basis_texture = pickle.load(open(self.pickle_basis_location, "rb"))
        else:
            self.basis_texture = None
        size = config.TERRAIN_SAMPLE_PATCH_SIZE
        # Check if the texture exists in a pickle, if so, load that
        self.pickle_texture_location = str.format("./engine/patches/{0}-{1}-texture.p", config.IMAGE_NAME,
                                                  world_transform.get_location_string())
        if os.path.isfile(self.pickle_texture_location):
            self.texture = pickle.load(open(self.pickle_texture_location, "rb"))
        else:
            self.texture = np.zeros((size, size, 3), dtype='uint8')

    # Basis texture is used for find_ssd
    def set_basis_texture(self, texture):
        self.basis_texture = texture
        with open(self.pickle_basis_location, 'wb') as f:
            pickle.dump(texture, f)

    def has_basis_texture(self):
        return self.basis_texture is not None

    def get_basis_filename(self):
        return self.pickle_basis_location
