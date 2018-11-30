import csv
import numpy as np


class Patient(object):
    age = 0
    sex = 0
    cp = 0
    trestbps = 0
    chol = 0
    fbs = 0
    restecg = 0
    thalach = 0
    exang = 0
    oldpeak = 0
    slope = 0
    ca = 0
    thal = 0
    num = 0

    # The class "constructor" - It's actually an initializer
    def __init__(self, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num):
       self.age = age
       self.sex = sex
       self.cp = cp
       self.trestbps = trestbps
       self.chol = chol
       self.fbs = fbs
       self.restecg = restecg
       self.thalach = thalach
       self.exang = exang
       self.oldpeak = oldpeak
       self.slope = slope
       self.ca = ca
       self.thal = thal
       self.num = num

    def __repr__(self):
        return "Patent Age: " + self.age + " Num: " + self.num


if __name__ == '__main__':
    with open('processed.cleveland.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        pats = []
        for row in csv_reader:
            pats.append(Patient(*row))
            line_count += 1
        print(f'Processed {line_count} lines.')
        patients = np.array(pats)
        print(patients)

    print("hello world")

