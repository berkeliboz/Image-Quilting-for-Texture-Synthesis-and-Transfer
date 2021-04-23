import pygame.image
import cv2 as cv
import numpy as np
from engine import display
from engine import entity_manager
from engine.entity import Entity
from engine.terrain import Terrain
from engine.transform import Transform
from pygame.math import Vector2
from game.hero import Hero
from game.bed import Bed

def init():
    entity_manager.EntityManager.getInstance()

    test_terrain = Terrain((100, 100), (Transform(Vector2(250, 250))), True)
    character = Hero((100,100), (Transform(Vector2(20, 20))), True)
    bed = Bed((100,100), (Transform(Vector2(120, 120))), True)


    main_display = display.Display()


if __name__ == "__main__":
    init()
