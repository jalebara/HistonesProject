from abc import ABCMeta, abstractmethod
import pandas as pd

class ModelPrototype(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, x:pd.DataFrame, y:pd.DataFrame ):
        pass

    @abstractmethod
    def predict(self, x:pd.DataFrame):
        pass
    
    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        """ A class is a ModelProtoype if it implements the fit and predict methods """
        return (hasattr(__subclass, 'fit') and callable(__subclass.fit) and
                hasattr(__subclass, 'predict') and callable(__subclass.predict))