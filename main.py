import cv2 as cv
import numpy as np
import random
import enum

path = 'textures/text.jpg'
number_of_blocks = 10
block_size = 40
random_sample_size = 500

# Numpy array
# Row Major

    # BGR

class Dir(enum.Enum):
   Up = 0
   Right = 1
   Both = 2

def main():
    img = cv.imread(path)

    patches = generate_path_list(img, 100)
    wow_image = create_image(patches, img)

    cv.imshow("mi",wow_image)
    cv.waitKey(0)

def create_patch_array(patches, img):
    arr = np.zeros((number_of_blocks,number_of_blocks,block_size,block_size, 3), dtype='uint8')
    for col in range(number_of_blocks):
        for row in range(number_of_blocks):
            if col == 0 and row == 0:
                arr[row,col] = find_ssd(patches[0], patches[0], generate_path_list(img, random_sample_size), Dir.Right)
            elif col == 0:
                arr[row,col] = (find_ssd(arr[row - 1][0],patches[0], generate_path_list(img, random_sample_size), Dir.Right))
            elif row == 0:
                arr[row,col] = (find_ssd(patches[0], arr[0][col - 1], generate_path_list(img, random_sample_size), Dir.Up))
            else:
                arr[row,col] = (find_ssd(arr[row - 1][col], arr[row][col - 1], generate_path_list(img, random_sample_size), Dir.Both))
    return arr
def create_image(patches, img):
    arr = create_patch_array(patches, img)
    block_px_size = block_size * number_of_blocks
    blank = np.zeros((block_px_size, block_px_size, 3), dtype='uint8')
    for block_index in range(number_of_blocks):
        block_start = block_index * block_size
        for block_index_y in range(number_of_blocks):
            block_start_y = block_index_y * block_size
            blank[block_start: block_start + block_size, block_start_y: block_start_y + block_size] = arr[block_index_y][block_index]
    return blank

def find_ssd(source_patch_left, source_patch_up, patches, direction):
    min_index = 0
    min_ssd = 10000000
    for index in range(len(patches)):
        if direction == Dir.Up:
            ssd = calculateSSD_Vertical(source_patch_up, patches[index], 1)
        elif direction == Dir.Right:
            ssd = calculateSSD_Horizontal(source_patch_left, patches[index], 1)
        else:
            ssd = calculateSSD_Both(source_patch_left,source_patch_up, patches[index], 1)

        if ssd < min_ssd:
            min_ssd = ssd
            min_index = index
    return patches[min_index]

def generate_path_list(img, number_of_patches):
    patches = []
    for index in range(number_of_patches):
        patches.append(grab_random_box(block_size, img))
    return patches

def calculateSSD_Horizontal(patch_left : np.ndarray,patch_right: np.ndarray, offset_px):
    #Check columns
                                # Right side
    patch_left_col = patch_left[:,np.arange(patch_right.shape[0] - offset_px, patch_right.shape[0])]
                                # Left side
    patch_right_col = patch_right[:,np.arange(0, offset_px)]
    return np.nansum((patch_left_col.astype("int") - patch_right_col.astype("int")) ** 2)

def calculateSSD_Vertical(patch_up : np.ndarray,patch_down: np.ndarray, offset_px):
    patch_up_col = patch_up[np.arange(patch_up.shape[0] - offset_px, patch_up.shape[0]), :]
    patch_down_col = patch_down[np.arange(0,offset_px), :]

    return np.nansum((patch_up_col.astype("int") - patch_down_col.astype("int")) ** 2)

def calculateSSD_Both(patch_left : np.ndarray,patch_up: np.ndarray,target_patch: np.ndarray, offset_px):
    patch_up_col = patch_up[np.arange(patch_up.shape[0] - offset_px, patch_up.shape[0]), :]
    patch_down_col = target_patch[np.arange(0,offset_px), :]

    patch_left_col = patch_left[:,np.arange(target_patch.shape[0] - offset_px, target_patch.shape[0])]
    patch_right_col = target_patch[:,np.arange(0, offset_px)]

    return (np.nansum((patch_up_col.astype("int") - patch_down_col.astype("int")) ** 2) +
            np.nansum((patch_left_col.astype("int") - patch_right_col.astype("int")) ** 2)) / 2

def generate_random_overlap(img):
    block_px_size = block_size * number_of_blocks
    blank = np.zeros((block_px_size,block_px_size,3), dtype='uint8')

    for block_index in range(number_of_blocks):
        block_start = block_index * block_size
        for block_index_y in range(number_of_blocks):
            block_start_y = block_index_y * block_size
            blank[block_start : block_start + block_size , block_start_y : block_start_y + block_size ] = grab_random_box(block_size, img)
    return blank

def grab_random_box(block_size, img: np.ndarray):
    row_start = random.randrange(0, img.shape[0] - block_size)
    column_start = random.randrange(0, img.shape[1] - block_size)
    return img[row_start : row_start+block_size , column_start : column_start + block_size]

def rescale(img: np.ndarray, scale = 0.1):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimensions = (width,height)
    return cv.resize(img, dimensions)

if __name__ == "__main__":
    main()