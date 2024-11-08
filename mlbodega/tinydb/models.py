from typing import Optional
from tinydb import TinyDB, where
from mlbodega.ports import Models as Collection
from mlbodega.schemas import Model, Experiment

class Models(Collection):
    def __init__(self, location: str, experiment: Experiment):
        self.db = TinyDB(f'{location}/database.json')
        self.table = self.db.table(f'models:{experiment.id}')
    
    def put(self, model: Model):
        self.table.upsert({**model.dump()}, where('hash') == model.hash)

    def get(self, hash: str) -> Optional[Model]:
        result = self.table.get((where('hash') == hash))
        return Model(**result) if result else None
    
    def list(self) -> list[Model]:
        return [Model(**result) for result in self.table.all()]
    
    def remove(self, model: Model):
        self.table.remove(where('hash') == model.hash)