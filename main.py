import sys
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException


try:
    a = 12/0
except ZeroDivisionError as e:
    raise KidneyException(e , sys)