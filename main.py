import cv2 as cv
import numpy as np
import random
import enum
import sys
from tqdm import tqdm

path = 'textures/text.jpg'
number_of_blocks = 8
block_size = 60
random_sample_size = 5000
offset = 5
min_error = 100000

alpha = 1

sample_img_path = 'images/al.jpg'
source_row_num = 10
source_col_num = 10

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
    #texture_transfer(img)
    patches = generate_path_list(img, random_sample_size)

    size = number_of_blocks* block_size
    final_image = np.zeros((size,size, 3), dtype='uint8')

    combine_all_horizontal(final_image, patches, number_of_blocks, number_of_blocks, None)

    do_vertical_cut_and_stich(final_image, number_of_blocks - 1)
    cv.imshow("last", final_image)
    cv.waitKey(0)

def texture_transfer(img):
    source_img = cv.imread(sample_img_path)
    source_img_row_size = int(source_img.shape[0] / source_row_num)
    source_img_col_size = int(source_img.shape[1] / source_col_num)
    transfer_img = np.zeros((source_row_num * source_img_row_size, source_col_num * source_img_col_size, source_img.shape[2]), dtype="uint8")

    random_patches_list = generate_path_list_precise(img,random_sample_size, source_img_row_size, source_img_col_size)

    combine_all_horizontal(transfer_img, random_patches_list, source_row_num, source_col_num, source_img)
    do_vertical_cut_and_stich(transfer_img, source_col_num - 1)

def calculate_correspondence_error(source_patch : np.ndarray,transfer_patch : np.ndarray):
    return calculateSSD_Vertical(source_patch, transfer_patch, transfer_patch.shape[0])

def do_vertical_cut_and_stich(image: np.ndarray, number_of_vertical_blocks):
    row = image.shape[0]
    col = image.shape[1]
    source_img_row_size = int(image.shape[0] / (number_of_vertical_blocks + 1))
    for row_value in range(number_of_vertical_blocks):

        middle_row = (row - source_img_row_size) - source_img_row_size * row_value
        up_patch = image[middle_row - offset : middle_row, :]
        down_patch = image[middle_row : middle_row + offset, :]
        combined = combine_patch_vertical(up_patch, down_patch)
        up = middle_row - offset
        down = middle_row
        image[middle_row - offset: row - offset, :] = image[middle_row: row, :]
        image[up:down , :] = combined
    crop_image = image[
                 0 :row - (number_of_vertical_blocks * offset),
                 0 :col - (number_of_vertical_blocks * offset)]
    cv.imshow("image", crop_image)

    cv.waitKey(0)
def combine_all_horizontal(image: np.ndarray, patches, number_of_rows, number_of_cols, source_img):
    left_patch = patches[0]
    row_shape_offset = left_patch.shape[0]
    col_shape_offset = left_patch.shape[1]
    up_patch = None
    source_img_row_size = 0
    source_img_col_size = 0
    transfer_patch = None
    if type(source_img) is np.ndarray:
        source_img_row_size = int(source_img.shape[0] / source_row_num)
        source_img_col_size = int(source_img.shape[1] / source_col_num)

    for row_line in range(number_of_rows):
        right_offset = 0
        height = 0
        last = None

        for col_line in range(number_of_cols - 1):
            print(row_line,col_line)
            if type(source_img) is np.ndarray:
                row_start = row_line * source_img_row_size
                col_start = col_line * source_img_col_size
                transfer_patch = source_img[row_start: row_start + source_img_row_size,
                                 col_start: col_start + source_img_col_size]
            if row_line == 0:
                right_patch = find_ssd(left_patch, left_patch, patches, Dir.Right, transfer_patch)
            else:
                prev_height = up_patch.shape[0] * (row_line-1)
                right_patch_offset = (up_patch.shape[1] - offset) * (col_line + 1)
                up_patch = image[prev_height: prev_height + row_shape_offset, right_patch_offset : right_patch_offset + col_shape_offset]

                right_patch = find_ssd(left_patch, up_patch, patches, Dir.Both, transfer_patch)
            combined = combine_patch_horizontal(left_patch, right_patch)

            last = np.concatenate((left_patch[:, 0:row_shape_offset - offset, :],
                                   combined,
                                   right_patch[:, offset:row_shape_offset, :]),
                                  axis=1)
            left_starting_point = right_offset
            right_offset += right_patch[:, offset:row_shape_offset, :].shape[1]

            height = last.shape[0] * row_line
            image[
                height : height + last.shape[0],
                left_starting_point : left_starting_point + last.shape[1]] = last
            last_col = left_starting_point + last.shape[1]
            left_patch = image[height : height + last.shape[0], last_col - col_shape_offset: last_col]
        right_offset = 0
        left_patch = image[height: height + row_shape_offset, 0 : col_shape_offset]
        up_patch = find_ssd(left_patch, left_patch, patches, Dir.Up, transfer_patch)
        left_patch = up_patch
        last_direction = Dir.Up
    cv.imshow("img",image)
    cv.waitKey(0)
    return
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
    offset_mask = offset_mask[:, :, 0].astype("uint8")
    offset_mask = 255 - offset_mask

    segment_top = up_patch[up_patch.shape[0] - offset: up_patch.shape[0], :]
    segment_bot = down_patch[0: offset, :]

    return create_mask_combination(segment_top, segment_bot, offset_mask)

def combine_patch_horizontal(left_patch, patch_right):
    offset_mask = (generate_ssd_matrix_horizontal(left_patch, patch_right))
    offset_mask = generate_path_mask_horizontal(offset_mask)
    offset_mask = offset_mask[:, :, 0].astype("uint8")
    offset_mask = 255 - offset_mask

    segment_left = left_patch[:, left_patch.shape[1] - offset: block_size]
    segment_right = patch_right[:, 0:offset]

    return create_mask_combination(segment_left, segment_right, offset_mask)

def generate_ssd_matrix_horizontal(patch_left: np.ndarray, patch_right: np.ndarray):
    offset_matrix = np.zeros((patch_left.shape[0],offset, 1), dtype= "long")
    left_row_size = patch_left.shape[1] # row

    for row in range(patch_left.shape[0]):
        for col in range(offset):
            err_b = int(patch_left[row, (left_row_size - offset) + col][0]) - int(patch_right[row,col][0])
            err_g = int(patch_left[row, (left_row_size - offset) + col][1]) - int(patch_right[row, col][1])
            err_r = int(patch_left[row, (left_row_size - offset) + col][2]) - int(patch_right[row, col][2])
            offset_matrix[row,col] = (err_r**2 + err_g**2 + err_b**2)/3.0
    return offset_matrix

def generate_ssd_matrix_vertical(patch_up: np.ndarray, patch_down: np.ndarray):
    offset_matrix = np.zeros((offset,patch_up.shape[1], 1), dtype= "long")
    up_row_size = patch_up.shape[0] # row

    for row in range(offset):
        for col in range(patch_up.shape[1]):
            err_b = int(patch_up[up_row_size - offset + row, col][0]) - int(patch_down[row, col][0])
            err_g = int(patch_up[up_row_size - offset + row, col][1]) - int(patch_down[row, col][1])
            err_r = int(patch_up[up_row_size - offset + row, col][2]) - int(patch_down[row, col][2])
            offset_matrix[row,col] = (err_r**2 + err_g**2 + err_b**2)/3.0
    return offset_matrix

def generate_path_mask_vertical(offset_matrix: np.ndarray):
    solution_matrix = np.zeros((offset_matrix.shape[0], offset_matrix.shape[1], offset_matrix.shape[2]), dtype="long")
    solution_matrix[:, 0] = offset_matrix[:, 0]
    for col in range(1, solution_matrix.shape[1]):
        for row in range(0, offset):
            up_left_row = row - 1 if row - 1 > 0 else 0
            down_left_row = row + 1 if row + 1 < offset else offset - 1
            min_val = min(solution_matrix[up_left_row : down_left_row + 1, col - 1])
            solution_matrix[row,col] = min_val + offset_matrix[row,col]

    last_col = solution_matrix.shape[1] - 1
    min_val = min(solution_matrix[:, last_col])
    last_row = np.where(solution_matrix[:, last_col] == min_val)[0][0]

    offset_matrix[0:last_row + 1, last_col] = 0
    offset_matrix[last_row + 1: offset, last_col] = 255

    sneaky_hack_counter = 0
    while(last_col > 0):
        result_dir = Dir.Both

        smallest_value = sys.maxsize
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
    return offset_matrix

def generate_path_mask_horizontal(offset_matrix: np.ndarray):
    solution_matrix = np.zeros((offset_matrix.shape[0],offset_matrix.shape[1], offset_matrix.shape[2]), dtype="long")
    solution_matrix[0,:] = offset_matrix[0,:]

    for row in range(1, solution_matrix.shape[0]):
        for col in range(0,offset):
            left_col = col - 1 if col - 1 > 0 else 0
            right_col = col + 1 if col + 1 < offset else offset - 1
            min_val = min(solution_matrix[row - 1, left_col : right_col + 1])
            solution_matrix[row,col] = min_val + offset_matrix[row,col]

    last_row = solution_matrix.shape[0] - 1
    min_val = min(solution_matrix[last_row, :])
    last_col = np.where(solution_matrix[last_row, :] == min_val)[0][0]

    offset_matrix[last_row, 0: last_col + 1] = 0
    offset_matrix[last_row, last_col + 1: offset] = 255

    sneaky_hack_counter = 0
    while(last_row > 0):
        result_dir = Dir.Both

        smallest_value = sys.maxsize
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

def find_ssd(source_patch_left, source_patch_up, patches, direction, transfer_patch):
    min_index = 0
    min_ssd = sys.maxsize
    correspondence_error = 1

    for index in range(len(patches)):
        ## Calculate transfer_patch_coefficient
        if type(transfer_patch) is np.ndarray:
            correspondence_error = calculateSSD_Vertical(patches[index], transfer_patch, transfer_patch.shape[0])

        if direction == Dir.Up:
            ssd = alpha * calculateSSD_Vertical(source_patch_up, patches[index], offset) + (1 - alpha) * correspondence_error
        elif direction == Dir.Right:
            ssd = alpha * calculateSSD_Horizontal(source_patch_left, patches[index], offset) + (1 - alpha) * correspondence_error
        else:
            ssd_both = calculateSSD_Both(source_patch_left,source_patch_up, patches[index], offset)
            ssd = alpha * ssd_both + (1 - alpha) * correspondence_error

        if ssd < min_ssd:
            if ssd > min_error:
                min_ssd = ssd
                min_index = index
    print(min_ssd)
    return patches[min_index]

def generate_path_list_precise(img, number_of_patches, row,col):
    patches = []
    for index in range(number_of_patches):
        patches.append(grab_random_box(img,row,col))
    return patches

def generate_path_list(img, number_of_patches):
    patches = []
    for index in range(number_of_patches):
        patches.append(grab_random_box(img,block_size,block_size))
    return patches

def calculateSSD_Horizontal(patch_left : np.ndarray,patch_right: np.ndarray, offset_px):
    #Check columns
                                # Right side
    patch_left_col = patch_left[:,np.arange(patch_right.shape[1] - offset_px, patch_right.shape[1])]
                                # Left side
    patch_right_col = patch_right[:,np.arange(0, offset_px)]
    return np.nansum((patch_left_col.astype("int") - patch_right_col.astype("int")) ** 2)

def calculateSSD_Vertical(patch_up : np.ndarray,patch_down: np.ndarray, offset_px):
    patch_up_col = patch_up[np.arange(patch_up.shape[0] - offset_px, patch_up.shape[0]), :]
    patch_down_col = patch_down[np.arange(0,offset_px), :]

    return np.nansum((patch_up_col.astype("int") - patch_down_col.astype("int")) ** 2)

def calculateSSD_Both(patch_left : np.ndarray,patch_up: np.ndarray,target_patch: np.ndarray, offset_px):
    ssd_vertical = int(calculateSSD_Vertical(patch_up, target_patch, offset_px))
    ssd_horizontal = int(calculateSSD_Horizontal(patch_left,target_patch, offset_px))
    return (ssd_vertical * ssd_horizontal)

def generate_random_overlap(img):
    block_px_size = block_size * number_of_blocks
    blank = np.zeros((block_px_size,block_px_size,3), dtype='uint8')

    for block_index in range(number_of_blocks):
        block_start = block_index * block_size
        for block_index_y in range(number_of_blocks):
            block_start_y = block_index_y * block_size
            blank[block_start : block_start + block_size , block_start_y : block_start_y + block_size ] = grab_random_box(img,block_size,block_size)
    return blank

def grab_random_box(img: np.ndarray, row_size, col_size):
    row_start = random.randrange(0, img.shape[0] - row_size)
    column_start = random.randrange(0, img.shape[1] - col_size)
    return img[row_start : row_start+row_size , column_start : column_start + col_size]

def rescale(img: np.ndarray, scale = 0.1):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimensions = (width,height)
    return cv.resize(img, dimensions)

if __name__ == "__main__":
    main()