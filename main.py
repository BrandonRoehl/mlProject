import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
import pandas as pd

if __name__ == '__main__':
    patients = pd.read_csv('processed.cleveland.data')
    print(patients.shape)
    print(patients.dtypes)
    # TODO maybe don't use the values
    # patients = patients.drop(axis=1)
    # patients = patients.apply(lambda x: x.fillna(0))


# 2, 6, 10, 13
