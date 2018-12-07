import numpy as np
from sklearn import preprocessing
from sklearn import compose
import pandas as pd

if __name__ == '__main__':
    patients = pd.read_csv('cleveland.data')
    patients = patients.take([2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57], axis=1)
    print(patients.shape)
    print(patients.dtypes)


# 2, 6, 10, 13
