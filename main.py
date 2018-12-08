from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.utils import to_categorical

if __name__ == '__main__':
    patients = pd.read_csv('processed.cleveland.data')
    print(patients.shape)
    print(patients.dtypes)
    # TODO maybe don't use the values
    patients = patients.apply(lambda x: x.fillna(0))

    column_length = len(patients.columns) - 1
    x = patients.columns[:column_length]
    y = patients.columns[column_length:]

    x_train, x_test, y_train, y_test = train_test_split(patients[x], patients[y], random_state=0)

    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(categories=[[0, 1],
                                                 [1, 2,3,4],
                                                 [0,1,2],
                                                 [1,2,3]]), [1, 2, 6, 10]),
         ("norm", Normalizer(norm='l1'), [0,3,4,5,7,8,9,11])])

    x_train = colT.fit_transform(x_train)
    x_test = colT.fit_transform(x_test)


    learning_rate = .001
    training_epochs = 5

    n_input = x_train.shape[1]
    n_output = 2 #y_train.shape[1]
    n_length = x_train.shape[0]
    n_hidden = 10
    n_hidden2 = 5

    # y_train = y_train.stack
    # y_test = y_test.stack
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # stack = y_train.val
    # print(stack.shape)
    # print(stack)

    model = tf.keras.Sequential()
    model.add(layers.Dense(n_hidden, activation='sigmoid', input_shape=(n_input,)))
    model.add(layers.Dense(n_hidden2, activation='sigmoid'))
    model.add(layers.Dense(n_output, activation='sigmoid'))
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs= training_epochs, batch_size=n_length, validation_data=(x_test, y_test), verbose=2)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss: ', score[0])
    print('Accuracy: ', score[1])



    # print(patients.shape)
# 1 2, 6, 10, 13
