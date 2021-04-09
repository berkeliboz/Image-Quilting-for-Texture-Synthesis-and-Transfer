import pygame.image
import cv2 as cv
import numpy as np
from engine import display
from engine import entity_manager
from engine.entity import Entity
from engine.terrain import Terrain
from engine.transform import Transform
from pygame.math import Vector2
import image_quilting

def init():
    entity_manager.EntityManager.getInstance()

    test_terrain = Terrain((500,500), (Transform(Vector2(0,0))))

    #test_entity = Entity((500,500), (Transform(Vector2(-100,-100))))

    #img = np.array(image_quilting.main())
    #rows, cols, dim = img.shape
    #M2 = np.float32([[0.5, 0, 0],
    #                 [0.1, 0.5, 0]])
    #M2[1, 2] = -M2[0, 1] * rows / 2
    #M2[0, 2] = -M2[1, 0] * cols / 2

    #sheared_img = cv.warpAffine(img, M2, (rows, cols))

    #test_entity.set_texture(sheared_img)

    test_entity2 = Entity((100, 100), (Transform(Vector2(100, 300))))
    img2 = np.array(cv.imread('textures/textures/g_bc4_00_color.png'))
    test_entity2.set_texture(img2)
    main_display = display.Display()

if __name__ == "__main__":
    init()