import os

import config
import image_quilting
import numpy as np
import pickle
import threading
import time

from engine.entity import Entity
from engine.transform import Transform
from direction import direction


class Terrain(Entity):
    def __init__(self, size, world_transform, enable_update):
        super().__init__(size, world_transform, enable_update)
        output_path = config.OUTPUT_PATH
        self.terrain_id_map = dict()

        self.patches_cache = dict()
        img = image_quilting.read_path_2RGB(config.SAMPLE_IMAGE_PATH)
        self.random_patches = image_quilting.generate_path_list(img, config.TERRAIN_SAMPLE_PATCH_SIZE)

        self.thread_lock = threading.Lock()
        self.texture_generator = Texture_generation_thread(self, config.TEXTURE_GENERATOR_MAX_LIMIT, config.TEXTURE_GENERATOR_MAX_LIMIT)
        self.texture_generator.start()

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

            patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform((row * (sample_patch_size - quilt_size), 0)),
                                  False)
            self.terrain_id_map[(row + 1, 0)] = patch
            if patch.has_basis_texture():
                # print("Used cached file " + patch.get_basis_filename())
                continue
            patch.set_basis_texture(result_texture)

    def fix_terrain(self, number_of_horizontal_patches, number_of_vertical_patches, row_offset, col_offset):
        patches = []
        for row in range(row_offset, number_of_vertical_patches + row_offset):
            for col in range(col_offset, number_of_horizontal_patches + col_offset - 1):
                while self.get_basis_texture_at(row, col) is None or self.get_basis_texture_at(row, col + 1) is None or self.get_texture_at(row + 1, col) is None:
                    time.sleep(0.2)
                texture = image_quilting.stitch_horizontal_lossy(self.get_basis_texture_at(row, col),
                                                                 self.get_basis_texture_at(row, col + 1))

                new_size = int(texture.shape[1] / 2)
                new_size_middle = int(new_size / 2)

                self.get_texture_at(row, col)[:, new_size_middle: new_size] = texture[:,
                                                                              new_size_middle: new_size_middle + new_size_middle]
                self.get_texture_at(row, col + 1)[:, 0: new_size_middle] = texture[:,
                                                                           new_size: new_size + new_size_middle]

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

class Terrain_Patch(Entity):
    def __init__(self, size, world_transform, enable_update):
        super().__init__(size, world_transform, False)
        serialize = config.ENABLE_GENERATOR_CACHING

        # Check if basis image exists in a pickle, if so, load that
        self.pickle_basis_location = str.format("./engine/patches/{0}-{1}-basis.p", config.IMAGE_NAME,
                                                world_transform.get_location_string())
        if os.path.isfile(self.pickle_basis_location) and serialize:
            self.basis_texture = pickle.load(open(self.pickle_basis_location, "rb"))
        else:
            self.basis_texture = None
        size = config.TERRAIN_SAMPLE_PATCH_SIZE
        # Check if the texture exists in a pickle, if so, load that
        self.pickle_texture_location = str.format("./engine/patches/{0}-{1}-texture.p", config.IMAGE_NAME,
                                                  world_transform.get_location_string())
        if os.path.isfile(self.pickle_texture_location) and serialize:
            self.texture = pickle.load(open(self.pickle_texture_location, "rb"))
        else:
            self.texture = np.zeros((size, size, 3), dtype='uint8')

    # Basis texture is used for find_ssd
    def set_basis_texture(self, texture):
        self.basis_texture = texture
        if config.ENABLE_GENERATOR_CACHING:
            with open(self.pickle_basis_location, 'wb') as f:
                pickle.dump(texture, f)

    def has_basis_texture(self):
        return self.basis_texture is not None

    def get_basis_filename(self):
        return self.pickle_basis_location

class Texture_generation_thread(threading.Thread):
    def __init__(self, terrain : Terrain, number_of_horizontal_patches, number_of_vertical_patches, start_row = 0, start_col = 0):
        self.start_row = start_row
        self.start_col = start_col

        self.terrain = terrain
        self.number_of_horizontal_patches = number_of_horizontal_patches
        self.number_of_vertical_patches = number_of_vertical_patches
        super().__init__(target= self.generate_terrain)


    def generate_terrain(self):
        quilt_size = config.TERRAIN_QUILTING_SIZE
        sample_patch_size = config.TERRAIN_SAMPLE_PATCH_SIZE

        # Case 0,0
        previous_texture = self.terrain.random_patches[0]
        patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform((0, 0)), False)

        self.terrain.thread_lock.acquire()
        self.terrain.terrain_id_map[(self.start_row, self.start_col)] = patch
        self.terrain.thread_lock.release()

        patch.set_basis_texture(previous_texture)

        # for row in range(self.start_row, self.start_row + self.number_of_vertical_patches):
        #     for col in range(self.start_col + 1, self.start_col + self.number_of_horizontal_patches):

        max_axis = 2 * max(self.number_of_vertical_patches, self.number_of_horizontal_patches)

        for diagonal_sum in range(1, max_axis):
            for row in range(0, diagonal_sum):
                col = diagonal_sum - row
                patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform(
                    (row * (sample_patch_size - quilt_size), col * (sample_patch_size - quilt_size))), False)

                self.terrain.thread_lock.acquire()
                self.terrain.terrain_id_map[(row, col)] = patch
                self.terrain.thread_lock.release()

                if patch.has_basis_texture():
                    # print("Used cached file " + patch.get_basis_filename())
                    continue

                if row == 0:
                    result_texture = image_quilting.find_ssd(self.terrain.get_basis_texture_at(row, col - 1), None,
                                                             self.terrain.random_patches, direction(False, True, False, False))
                else:
                    horizontal = self.terrain.get_basis_texture_at(row, col - 1)
                    vertical = self.terrain.get_basis_texture_at(row - 1, col)
                    result_texture = image_quilting.find_ssd(horizontal, vertical, self.terrain.random_patches,
                                                             direction(True, True, False, False))
                patch.set_basis_texture(result_texture)

            # Case 0 N, grab first texture of previous row
            result_texture = image_quilting.find_ssd(None, self.terrain.get_basis_texture_at(row, 0), self.terrain.random_patches,
                                                     direction(True, False, False, False))

            patch = Terrain_Patch((sample_patch_size, sample_patch_size), Transform((row * (sample_patch_size - quilt_size), 0)),
                                  False)

            self.terrain.thread_lock.acquire()
            self.terrain.terrain_id_map[(row + 1, 0)] = patch
            self.terrain.thread_lock.release()

            if patch.has_basis_texture():
                # print("Used cached file " + patch.get_basis_filename())
                continue
            patch.set_basis_texture(result_texture)

class Quilting_thread(threading.Thread):
    def __init__(self, terrain : Terrain, row_offset,col_offset, number_of_horizontal_patches,number_of_vertical_patches, quilted_patches = None):
        self.quilted_patches = {}
        self.buffer = []
        self.terrain = terrain
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.number_of_horizontal_patches = number_of_horizontal_patches
        self.number_of_vertical_patches = number_of_vertical_patches
        self.current_row = 0
        self.current_col = 0
        self.kill = False
        super().__init__(target= self.quilt_area)

    def add_to_buffer(self, patch):
        self.terrain.thread_lock.acquire()
        self.buffer.append(patch)
        self.terrain.thread_lock.release()

    def pop_buffer(self):
        self.terrain.thread_lock.acquire()
        buffer_cpy = self.buffer.copy()
        self.buffer = []
        self.terrain.thread_lock.release()
        return buffer_cpy

    def update_current_draw_start(self,row,col):
        if (abs(self.col_offset - col) > 3):
            self.col_offset = col
            # print("col updated")
        if (abs(self.row_offset - row) > 3):
            self.row_offset = row
            # print("row updated")

    def quilt_area(self):
        self.draw_area()

    def draw_area(self):
        for row in range(self.row_offset, self.number_of_horizontal_patches + self.row_offset):
             for col in range(self.col_offset, self.number_of_vertical_patches + self.col_offset - 1):
                self.current_col = col
                self.current_row = row
                while self.terrain.get_basis_texture_at(row, col) is None or self.terrain.get_basis_texture_at(row, col + 1) is None or self.terrain.get_texture_at(row + 1, col) is None:
                    time.sleep(0.2)
                if self.quilted_patches.get((row,col)):
                    continue
                texture = image_quilting.stitch_horizontal_lossy(self.terrain.get_basis_texture_at(row, col),
                                                                 self.terrain.get_basis_texture_at(row, col + 1))
                new_size = int(texture.shape[1] / 2)
                new_size_middle = int(new_size / 2)
                self.terrain.thread_lock.acquire()
                self.terrain.get_texture_at(row, col)[:, new_size_middle: new_size] = texture[:,
                                                                              new_size_middle: new_size_middle + new_size_middle]
                self.terrain.get_texture_at(row, col + 1)[:, 0: new_size_middle] = texture[:,
                                                                           new_size: new_size + new_size_middle]
                self.terrain.thread_lock.release()

                image = image_quilting.stitch_vertical_lossy(self.terrain.get_texture_at(row, col),
                                                             self.terrain.get_texture_at(row + 1, col))
                new_size = int(image.shape[0] / 2)
                new_size_middle = int(new_size / 2)
                self.terrain.thread_lock.acquire()
                self.terrain.get_texture_at(row, col)[new_size_middle: new_size, :] = image[
                                                                              new_size_middle: new_size_middle + new_size_middle,
                                                                              :]
                self.terrain.get_texture_at(row + 1, col)[0: new_size_middle, :] = image[
                                                                           new_size: new_size + new_size_middle,
                                                                           :]
                self.quilted_patches[(row, col)] = self.terrain.get_patch(row, col)
                self.terrain.thread_lock.release()
                self.add_to_buffer(self.terrain.get_patch(row, col))

                if self.kill:
                    if self.terrain.thread_lock.locked():
                        self.terrain.thread_lock.release()
                    return self.terrain.get_patch(row, col)