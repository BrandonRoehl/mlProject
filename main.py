import numpy as np
from sklearn import preprocessing

if __name__ == '__main__':
    patients = np.genfromtxt('cleveland.data', delimiter=' ')
    patients_decreased = np.delete(patients,[0,1,4,5,6,7,10,12,13,14,16,17,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,38,41,42,44,45,46,47,48,49,51,52,53,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74],1)
    print(patients_decreased)

    # patients = patients[~np.isnan(patients).any(axis=1)]
    # print(patients[0])
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # oneHot = preprocessing.OneHotEncoder(categorical_features=[2,6,10,13])
    #
    # patients_encoded = oneHot.fit_transform(patients).toarray()
    #
    # patients_scaled = min_max_scaler.fit_transform(patients_encoded)
    # print(patients_scaled[0])
    # patients_scaled = np.array(patients_scaled)


    # print(np.delete(patients_scaled, [13], 1)[0])
    # patients_scaled = np.delete(patients_scaled, [13], 1)[0]

    # print(patients)

    print("hello world")



# 2, 6, 10, 13