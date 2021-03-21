import cv2 as cv
import numpy as np
import random
import enum
from tqdm import tqdm

path = 'textures/apples.png'
number_of_blocks = 4
block_size = 100
random_sample_size = 100
offset = 15

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
   Down = 4


def main():
    img = cv.imread(path)

#    test_1 = cv.imread(test_1_path)
#    test_2 = cv.imread(test_2_path)

    patches = generate_path_list(img, random_sample_size)

    left_patch = patches[0]
    random_good_patch = find_ssd(patches[0], patches[0], generate_path_list(img, random_sample_size), Dir.Up)

    size = number_of_blocks* block_size
    final_image = np.zeros((size,size, 3), dtype='uint8')
    image, arr = create_image(patches, img)
    combine_all_vertical(final_image, arr)
    cv.imshow("he", image)
    cv.waitKey(0)

def combine_all_vertical(image: np.ndarray, arr: np.ndarray):
    size_vertical = arr.shape[0]
    size_horizontal = arr.shape[1]

    left_patch = arr[0][0]

    image[0: 0 + left_patch.shape[0], 0: int(left_patch.shape[0]/2)] = left_patch[:, 0: int(left_patch.shape[0]/2)]
    left_start_offset = int(left_patch.shape[0]/2)


    for col_line in range(size_vertical):
        right_offset = 0
        for row_line in range(size_horizontal - 1):
            left_patch = arr[row_line][col_line]
            right_patch = arr[row_line + 1][col_line]

            combined = combine_patch_horizontal(arr[row_line][col_line], arr[row_line + 1][col_line])
            last = np.concatenate((left_patch[:, 0:block_size - offset, :],
                                   combined,
                                   right_patch[:, offset:block_size, :]),
                                  axis=1)
            cv.imshow("last", last)

            left_starting_point = right_offset
            right_offset += right_patch[:, offset:block_size, :].shape[1]

            image[
                0 : 0 + last.shape[0],
                left_start_offset + left_starting_point : left_starting_point + last.shape[1]] = last[:, left_start_offset : last.shape[1]]
            cv.imshow("mi", image)
            cv.imshow("added", last[:, left_start_offset: last.shape[1]])

            cv.waitKey(0)
def create_mask_combination(segment_top, segment_bot, offset_mask):
    segment_left_b, segment_left_g, segment_left_r = cv.split(segment_top)
    segment_left_b = segment_left_b.astype("uint8")
    segment_left_g = segment_left_g.astype("uint8")
    segment_left_r = segment_left_r.astype("uint8")

    segment_right_b, segment_right_g, segment_right_r = cv.split(segment_bot)
    segment_right_b = segment_right_b.astype("uint8")
    segment_right_g = segment_right_g.astype("uint8")
    segment_right_r = segment_right_r.astype("uint8")

    result_leftside_b = cv.bitwise_and(segment_left_b, offset_mask)
    result_leftside_g = cv.bitwise_and(segment_left_g, offset_mask)
    result_leftside_r = cv.bitwise_and(segment_left_r, offset_mask)
    offset_mask = 255 - offset_mask
    result_rightside_b = cv.bitwise_and(segment_right_b, offset_mask)
    result_rightside_g = cv.bitwise_and(segment_right_g, offset_mask)
    result_rightside_r = cv.bitwise_and(segment_right_r, offset_mask)
    combined_b = result_leftside_b + result_rightside_b
    combined_g = result_leftside_g + result_rightside_g
    combined_r = result_leftside_r + result_rightside_r

    return cv.merge([combined_b, combined_g, combined_r])
def combine_patch_vertical(up_patch: np.ndarray,down_patch: np.ndarray):
    offset_mask = (generate_ssd_matrix_vertical(up_patch, down_patch))
    offset_mask = generate_path_mask_vertical(offset_mask)
    offset_mask = 255 - offset_mask

    segment_top = up_patch[block_size - offset: block_size, :]
    segment_bot = down_patch[0: offset, :]

    return create_mask_combination(segment_top, segment_bot, offset_mask)

def combine_patch_horizontal(left_patch, patch_right):
    offset_mask = (generate_ssd_matrix_horizontal(left_patch, patch_right))
    offset_mask = generate_path_mask_horizontal(offset_mask)
    offset_mask = offset_mask[:, :, 0].astype("uint8")
    offset_mask = 255 - offset_mask

    segment_left = left_patch[:, block_size - offset: block_size]
    segment_right = patch_right[:, 0:offset]

    return create_mask_combination(segment_left, segment_right, offset_mask)

def create_patch_array(patches, img):
    progress_bar = tqdm(range(number_of_blocks * number_of_blocks))
    progress_bar.set_description("Finding patch arrays",refresh=False)
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
    return blank, arr

def generate_ssd_matrix_horizontal(patch_left: np.ndarray, patch_right: np.ndarray):
    offset_matrix = np.zeros((block_size,offset, 1), dtype= "long")
    left_row_size = patch_left.shape[0] # row

    for row in range(block_size):
        for col in range(offset):
            err_b = int(patch_left[row, (left_row_size - offset) + col][0]) - int(patch_right[row,col][0])
            err_g = int(patch_left[row, (left_row_size - offset) + col][1]) - int(patch_right[row, col][1])
            err_r = int(patch_left[row, (left_row_size - offset) + col][2]) - int(patch_right[row, col][2])
            offset_matrix[row,col] = (err_r**2 + err_g**2 + err_b**2)/3.0
    return offset_matrix

def generate_ssd_matrix_vertical(patch_up: np.ndarray, patch_down: np.ndarray):
    offset_matrix = np.zeros((offset,block_size, 1), dtype= "long")
    up_row_size = patch_up.shape[0] # row

    for row in range(offset):
        for col in range(block_size):
            err_b = int(patch_up[up_row_size - offset + row, col][0]) - int(patch_down[row, col][0])
            err_g = int(patch_up[up_row_size - offset + row, col][1]) - int(patch_down[row, col][1])
            err_r = int(patch_up[up_row_size - offset + row, col][2]) - int(patch_down[row, col][2])
            offset_matrix[row,col] = (err_r**2 + err_g**2 + err_b**2)/3.0
    return offset_matrix

def generate_path_mask_vertical(offset_matrix: np.ndarray):
    solution_matrix = np.zeros((offset_matrix.shape[0], offset_matrix.shape[1], offset_matrix.shape[2]), dtype="long")
    solution_matrix[:, 0] = offset_matrix[:, 0]
    for col in range(1, block_size):
        for row in range(0, offset):
            up_left_row = row - 1 if row - 1 > 0 else 0
            down_left_row = row + 1 if row + 1 < offset else offset - 1
            min_val = min(solution_matrix[up_left_row : down_left_row + 1, col - 1])
            solution_matrix[row,col] = min_val + offset_matrix[row,col]

    last_col = block_size - 1
    min_val = min(solution_matrix[:, last_col])
    last_row = np.where(solution_matrix[:, last_col] == min_val)[0][0]

    offset_matrix[0:last_row + 1, last_col] = 0
    offset_matrix[last_row + 1: offset, last_col] = 255

    sneaky_hack_counter = 0
    while(last_col > 0):
        result_dir = Dir.Both

        smallest_value = 100000000000
        result_row = 0
        result_col = 0
        if last_row - 1 >= 0:
            if solution_matrix[last_row - 1,last_col] < smallest_value:
                result_row, result_col = last_row - 1, last_col
                smallest_value = solution_matrix[last_row - 1,last_col]
                result_dir = Dir.Up
        if last_row + 1 < offset:
            if solution_matrix[last_row + 1, last_col] < smallest_value:
                result_row, result_col = last_row + 1, last_col
                smallest_value = solution_matrix[last_row + 1, last_col]
                result_dir = Dir.Down
        if last_col > 0:
            if solution_matrix[last_row, last_col - 1] < smallest_value:
                result_row, result_col = last_row, last_col - 1
                smallest_value = solution_matrix[last_row, last_col - 1]
                result_dir = Dir.Left
        last_row, last_col = result_row, result_col
        if result_dir == Dir.Up:
            if sneaky_hack_counter > 3:
                last_row, last_col = last_row, last_col - 1
                offset_matrix[0: last_row + 1, last_col + 1] = 0
                offset_matrix[last_row + 1: offset, last_col + 1] = 255
                sneaky_hack_counter = False
        if result_dir == Dir.Down:
            sneaky_hack_counter+= 1

        if result_dir == Dir.Left:
            sneaky_hack_counter = 0
            offset_matrix[0 : last_row + 1, last_col+1 ] = 0
            offset_matrix[last_row + 1 : offset,last_col+1] = 255
    offset_matrix[0: last_row + 1, last_col ] = 0
    offset_matrix[last_row + 1:offset, last_col] = 255
    offset_matrix = offset_matrix[:, :, 0].astype("uint8")
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
        if result_dir == Dir.Right:
            if sneaky_hack_counter > 3:
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