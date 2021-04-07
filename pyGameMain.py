import pygame.image
import cv2 as cv
import numpy as np
from engine import display
from engine import entity_manager
from engine import entity


def init():
    entity_manager.EntityManager.getInstance()

    test_entity = entity.Entity((100,100))
    img = np.array(cv.imread('textures/textures/g_wt2_00_color.png'))

    rows, cols, dim = img.shape
    M2 = np.float32([[0.5, 0, 0],
                     [0.1, 0.5, 0]])
    M2[1, 2] = -M2[0, 1] * rows / 2
    M2[0, 2] = -M2[1, 0] * cols / 2
    sheared_img = cv.warpAffine(img, M2, (rows, cols))

    test_entity.set_texture(sheared_img)
    main_display = display.Display()

if __name__ == "__main__":
    init()