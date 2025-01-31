import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from thyroid_disease_det.exception import thyroid_disease_detException
from thyroid_disease_det.logger import logging


#class BinaryValueMapping:
#      self.f: int = 0
#        self.t: int = 1
#        self.F: int = 0
#        self.M: int = 1
#
#    def _asdict(self):
#        return self.__dict__
#
#    def reverse_mapping(self):
#        mapping_response = self._asdict()
#        return dict(zip(mapping_response.values(), mapping_response.keys()))  

class TargetValueMapping:
    def __init__(self):
        self.negative = 0
        self.compensated_hypothyroid = 1
        self.primary_hypothyroid = 2
        self.secondary_hypothyroid = 3

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
             