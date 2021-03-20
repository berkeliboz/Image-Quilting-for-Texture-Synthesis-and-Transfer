import cv2 as cv
import numpy as np
import random
import enum
from tqdm import tqdm

path = 'textures/apples.png'
number_of_blocks = 10
block_size = 100
random_sample_size = 500
offset = 10

test_1_path = 'textures/test_l.png'
test_2_path = 'textures/test_r.png'
# Numpy array
# Row Major

    # BGR

class Dir(enum.Enum):
   Up = 0
   Right = 1
   Both = 2
   Left = 3


def main():
    img = cv.imread(path)

#    test_1 = cv.imread(test_1_path)
#    test_2 = cv.imread(test_2_path)

    patches = generate_path_list(img, random_sample_size)

    left_patch = patches[0]
    random_good_patch = find_ssd(patches[0], patches[0], generate_path_list(img, random_sample_size), Dir.Right)

    offset_mask = (generate_ssd_matrix_horizontal(left_patch, random_good_patch))
    offset_mask = generate_path_mask_horizontal(offset_mask)
    offset_mask = offset_mask[:,:,0].astype("uint8")
    offset_mask = 255 - offset_mask

    segment_left = left_patch[:, block_size - offset: block_size]
    segment_right = random_good_patch[:, 0:offset]

    segment_left_b,segment_left_g,segment_left_r = cv.split(segment_left)
    segment_left_b = segment_left_b.astype("uint8")
    segment_left_g = segment_left_g.astype("uint8")
    segment_left_r = segment_left_r.astype("uint8")

    segment_right_b, segment_right_g, segment_right_r = cv.split(segment_right)
    segment_right_b = segment_right_b.astype("uint8")
    segment_right_g = segment_right_g.astype("uint8")
    segment_right_r = segment_right_r.astype("uint8")

    result_leftside_b = cv.bitwise_and(segment_left_b,offset_mask)
    result_leftside_g = cv.bitwise_and(segment_left_g,offset_mask)
    result_leftside_r = cv.bitwise_and(segment_left_r,offset_mask)
    offset_mask = 255 - offset_mask
    result_rightside_b = cv.bitwise_and(segment_right_b, offset_mask)
    result_rightside_g = cv.bitwise_and(segment_right_g, offset_mask)
    result_rightside_r = cv.bitwise_and(segment_right_r, offset_mask)
    combined_b = result_leftside_b + result_rightside_b
    combined_g = result_leftside_g + result_rightside_g
    combined_r = result_leftside_r + result_rightside_r

    combined = cv.merge([combined_b,combined_g,combined_r])

    # wow_image = create_image(patches, img)
    # cv.imshow("he", wow_image)
    # cv.waitKey(0)
    cv.imshow("mi",np.concatenate((left_patch[:,0:block_size - offset,:], combined, random_good_patch[:,offset:block_size,:]), axis=1))
    cv.waitKey(0)
    cv.imshow('hi', offset_mask)
    cv.waitKey(0)

#def quilt_vertical(base : np.ndarray, mask: np.ndarray, dst : np.ndarray):

def create_patch_array(patches, img):
    current = 0
    progress_bar = tqdm(range(number_of_blocks * number_of_blocks))
    progress_bar.set_description("Creating patch arrays",refresh=False)
    for current in progress_bar:
        arr = np.zeros((number_of_blocks,number_of_blocks,block_size,block_size, 3), dtype='uint8')
        for col in range(number_of_blocks):
            for row in range(number_of_blocks):
                current+=1
                if col == 0 and row == 0:
                    arr[row,col] = find_ssd(patches[0], patches[0], patches, Dir.Right)
                elif col == 0:
                    arr[row,col] = (find_ssd(arr[row - 1][0],patches[0], patches, Dir.Right))
                elif row == 0:
                    arr[row,col] = (find_ssd(patches[0], arr[0][col - 1], patches, Dir.Up))
                else:
                    arr[row,col] = (find_ssd(arr[row - 1][col], arr[row][col - 1], patches, Dir.Both))
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

def generate_ssd_matrix_horizontal(patch_left: np.ndarray, patch_right: np.ndarray):
    offset_matrix = np.zeros((block_size,offset, 1), dtype= "long")
    left_row_size = patch_left.shape[0] # row

    for row in range(block_size):
        for col in range(offset):
            left = patch_left[row, (left_row_size - offset) + col].astype("int")
            right = patch_right[row,col].astype("int")
            val = abs((left - right  ** 2)).astype("int")
            energy = val[0] + val[1] + val[2]
            offset_matrix[row,col] = energy
    return offset_matrix

def generate_path_mask_horizontal_old(offset_matrix: np.ndarray):
    previous_index = 0
    for row in range(block_size):
        if row == 0:
            min_index = 0
            min_energy = 1000000
            for col in range(offset):
                if offset_matrix[row,col] < min_energy:
                    min_energy = offset_matrix[row,col]
                    min_index = col
            for col in range(offset):
                if col == min_index:
                    previous_index = col
                offset_matrix[row,col] = 255 if min_index <= col else 0
        else:
            left_index = previous_index - 1 if previous_index > 0 else 0
            right_index = previous_index + 1 if previous_index + 1 < offset else offset - 1

            min_val = min(offset_matrix[row, left_index : right_index])
            for index in range(left_index, right_index):
                if offset_matrix[row, index] == min_val:
                    offset_matrix[row, 0 : index] = 0
                    offset_matrix[row, index:offset] = 255
                    previous_index = index
                    print(previous_index)
                    break
    return offset_matrix

def generate_path_mask_horizontal(offset_matrix: np.ndarray):
    solution_matrix = np.zeros((offset_matrix.shape[0],offset_matrix.shape[1], offset_matrix.shape[2]), dtype="long")
    solution_matrix[0,:] = offset_matrix[0,:]

    for row in range(1,block_size):
        for col in range(0,offset):
            left_col = col - 1 if col - 1 > 0 else 0
            right_col = col + 1 if col + 1 < offset else offset - 1
            min_val = min(solution_matrix[row - 1, left_col : right_col + 1])
            solution_matrix[row,col] = min_val + offset_matrix[row,col]

    last_row = block_size - 1
    min_val = min(solution_matrix[last_row, :])
    last_col = np.where(solution_matrix[last_row, :] == min_val)[0][0]

    offset_matrix[last_row, 0: last_col + 1] = 0
    offset_matrix[last_row, last_col + 1: offset] = 255

    sneaky_hack_counter = 0

    while(last_row > 0):
        print(last_row, last_col)
        result_dir = Dir.Both

        smallest_value = 100000000000
        result_row = 0
        result_col = 0
        if last_col - 1 > 0:
            if solution_matrix[last_row,last_col - 1] < smallest_value:
                result_row, result_col = last_row, last_col - 1
                smallest_value = solution_matrix[last_row,last_col - 1]
                result_dir = Dir.Left
        if last_col + 1 < offset:
            if solution_matrix[last_row, last_col + 1] < smallest_value:
                result_row, result_col = last_row, last_col + 1
                smallest_value = solution_matrix[last_row, last_col + 1]
                result_dir = Dir.Right

        if last_row > 0:
            if solution_matrix[last_row - 1, last_col] < smallest_value:
                result_row, result_col = last_row - 1, last_col
                smallest_value = solution_matrix[last_row - 1, last_col]
                result_dir = Dir.Up
        last_row, last_col = result_row, result_col
        print(result_dir)
        if result_dir == Dir.Right:
            if sneaky_hack_counter > 10:
                last_row, last_col = last_row - 1, last_col
                offset_matrix[last_row + 1, 0: last_col + 1] = 0
                offset_matrix[last_row + 1, last_col + 1:offset] = 255
                sneaky_hack_counter = False
        if result_dir == Dir.Left:
            sneaky_hack_counter+= 1
        if result_dir == Dir.Up:
            sneaky_hack_counter = 0
            offset_matrix[last_row + 1, 0: last_col + 1] = 0
            offset_matrix[last_row + 1, last_col + 1:offset] = 255


    offset_matrix[last_row, 0: last_col + 1] = 0
    offset_matrix[last_row, last_col + 1:offset] = 255
    return offset_matrix

def find_ssd(source_patch_left, source_patch_up, patches, direction):
    min_index = 0
    min_ssd = 10000000
    for index in range(len(patches)):
        if direction == Dir.Up:
            ssd = calculateSSD_Vertical(source_patch_up, patches[index], offset)
        elif direction == Dir.Right:
            ssd = calculateSSD_Horizontal(source_patch_left, patches[index], offset)
        else:
            ssd = calculateSSD_Both(source_patch_left,source_patch_up, patches[index], offset)

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
    offset_px = int(offset_px/2)
                                # Right side
    patch_left_col = patch_left[:,np.arange(patch_right.shape[0] - offset_px, patch_right.shape[0])]
                                # Left side
    patch_right_col = patch_right[:,np.arange(0, offset_px)]
    return np.nansum((patch_left_col.astype("int") - patch_right_col.astype("int")) ** 2)

def calculateSSD_Vertical(patch_up : np.ndarray,patch_down: np.ndarray, offset_px):
    offset_px = int(offset_px/2)
    patch_up_col = patch_up[np.arange(patch_up.shape[0] - offset_px, patch_up.shape[0]), :]
    patch_down_col = patch_down[np.arange(0,offset_px), :]

    return np.nansum((patch_up_col.astype("int") - patch_down_col.astype("int")) ** 2)

def calculateSSD_Both(patch_left : np.ndarray,patch_up: np.ndarray,target_patch: np.ndarray, offset_px):
    offset_px = int(offset_px/2)
    ssd_vertical = calculateSSD_Vertical(patch_up, target_patch, offset_px)
    ssd_horizontal = calculateSSD_Horizontal(patch_left,target_patch, offset_px)
    return (ssd_vertical + ssd_horizontal) / 2

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