import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
import pandas as pd

if __name__ == '__main__':
    patients = pd.read_csv('processed.cleveland.data')
    print(patients.shape)
    print(patients.dtypes)
    # TODO maybe don't use the values
    patients = patients.apply(lambda x: x.fillna(0))


    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(categories=[[0, 1],
                                                 [1, 2,3,4],
                                                 [0,1,2],
                                                 [1,2,3],
                                                 [0,1]]), [1, 2, 6, 10, 13]),
         ("norm", Normalizer(norm='l1'), [0,3,4,5,7,8,9,11])])

    patients = colT.fit_transform(patients)
    print(patients)
# 1 2, 6, 10, 13
