from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import activations


if __name__ == '__main__':
    # Hyper parameters
    learning_rate = 0.001
    decay_rate = 0.0002
    training_epochs = 335
    display_epochs = 100
    batch_size = 100
    hidden_nodes = 72
    data_type = 'float'

    # Read in the processed data
    patients = pd.read_csv('processed.cleveland.data', dtype='object', header=None, names=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'num'
    ])
    # Sanitize based on incorrect data
    for x in patients.columns:
        patients = patients[patients[x] != '?']
        patients = patients[patients[x] != '-9']
    # Convert to a float matrix
    patients = patients.astype(data_type)
    # Only train on patients with have or don't
    patients = patients[patients.num <= 1]

    print(patients.shape)
    # => (212 , 14)
    print(patients.dtypes)
    # => float64

    # Split into test and train data
    x = patients.drop(['num'], axis=1)
    y = patients.take([13], axis=1)

    # Normalize X data
    colT = ColumnTransformer(
        [("onehot", OneHotEncoder(categories=[[0, 1],
                                              [1, 2, 3, 4],
                                              [0, 1],
                                              [0, 1, 2],
                                              [0, 1],
                                              [1,2,3],
                                              [0,1,2,3],
                                              [3,6,7]]), [1,2,5,6,8,10,11,12]),
         ("norm", Normalizer(norm='l1'), [0,3,4,7,9])])

    x = colT.fit_transform(x)

    # Normalize Y data
    y = to_categorical(y)

    # Split the data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # Determined parameters
    n_input = x_train.shape[1]
    n_output = y_train.shape[1]

    input_layer = layers.Input(shape=(n_input,))
    hidden_layers = layers.Dense(hidden_nodes, activation=activations.elu)(input_layer)
    output_layer = layers.Dense(n_output, activation=activations.sigmoid)(hidden_layers)

    model = Model(input_layer, output_layer)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate, decay=decay_rate), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              verbose=2)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss: ', score[0])
    print('Accuracy: ', score[1])
