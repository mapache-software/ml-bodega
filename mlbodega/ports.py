from abc import ABC, abstractmethod
from typing import Optional

from mlbodega.schemas import Experiment
from mlbodega.schemas import Model

class Experiments(ABC):
    @abstractmethod
    def create(self, name: str) -> Experiment: ...

    @abstractmethod
    def list(self) -> list[Experiment]: ...

    @abstractmethod
    def read(self, name: str) -> Optional[Experiment]: ...

    @abstractmethod
    def update(self, experiment: Experiment): ...

    @abstractmethod
    def delete(self, name: str): ...
    

class Models(ABC):
    experiment: Experiment

    @abstractmethod
    def list(self) -> list[Model]: ...

    @abstractmethod
    def get(self, hash: str) -> Optional[Model]: ...

    @abstractmethod
    def put(self, model: Model): ...

    @abstractmethod
    def remove(self, model: Model): ...