import engine.entity as Entity
import heapq

class EntityManager():
    __instance = None
    entity_map = dict()
    entity_heap = []
    free_ids = list()
    max_id = 0

    @staticmethod
    def getInstance():
        if EntityManager.__instance == None:
            EntityManager()
        return EntityManager.__instance

    def __init__(self):
        if EntityManager.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            EntityManager.__instance = self

    @staticmethod
    def free_id(id):
        EntityManager.free_ids.append(id)

    @staticmethod
    def generate_id():
        if len(EntityManager.free_ids) == 0:
            value = EntityManager.max_id
            EntityManager.max_id +=1
            return value
        return EntityManager.free_ids.pop()

    @staticmethod
    def register_entity(entity : Entity):
        new_id = EntityManager.generate_id()
        EntityManager.entity_map[new_id] = entity
        heapq.heappush(EntityManager.entity_heap, (new_id, entity))
        return new_id