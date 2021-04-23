import pygame.image
import cv2 as cv
import numpy as np
from engine import display
from engine import entity_manager
from engine.entity import Entity
from engine.terrain import Terrain
from engine.transform import Transform
from pygame.math import Vector2

from game.chest import Chest
from game.crate_1 import Crate1
from game.crate_2 import Crate2
from game.cupboard import Cupboard
from game.hero import Hero
from game.bed import Bed
from game.stove import Stove
from game.table import TableO
from game.walls import Walls


def init():
    entity_manager.EntityManager.getInstance()

    test_terrain = Terrain((100, 100), (Transform(Vector2(250, 250))), True)
    walls = Walls()
    character = Hero((100, 100), (Transform(Vector2(20, 20))), True)
    bed = Bed((100, 100), (Transform(Vector2(120, 120))), True)
    table = TableO((100, 100), (Transform(Vector2(612, 180))), True)
    crate_1 = Crate1((100, 100), (Transform(Vector2(720, 540))), True)
    crate_2 = Crate2((100, 100), (Transform(Vector2(288, 324))), True)
    stove = Stove((100, 100), (Transform(Vector2(864, 36))), True)
    cupboard = Cupboard((100, 100), (Transform(Vector2(72, 504))), True)
    chest = Chest((100, 100), (Transform(Vector2(504, 576))), True)

    main_display = display.Display()


if __name__ == "__main__":
    init()
