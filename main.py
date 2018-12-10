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
    learning_rate = .001
    training_epochs = 2000
    display_epochs = 100
    hidden_nodes = 72
    batch_size = 10000
    data_type = 'float'

    # Read in the processed data
    patients = pd.read_csv('processed.cleveland.data', dtype='object', header=None, names=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'depression',
        'exercise', 'ca', 'thal'
    ])
    # Sanitize based on incorrect data
    for x in patients.columns:
        patients = patients[patients[x] != '?']
        patients = patients[patients[x] != '-9']
    # Convert to a float matrix
    patients = patients.astype(data_type)
    # Only train on patients with have or don't
    patients = patients[patients.thal <= 1]

    print(patients.shape)
    # => (212 , 14)
    print(patients.dtypes)
    # => float64

    # Split into test and train data
    x = patients.drop(['thal'], axis=1)
    y = patients.take([13], axis=1)

    # Normalize X data
    colT = ColumnTransformer(
        [("onehot", OneHotEncoder(categories=[[0, 1],
                                              [1, 2, 3, 4],
                                              [0, 1, 2],
                                              [1, 2, 3]]), [1, 2, 6, 10]),
         ("norm", Normalizer(norm='l1'), [0, 3, 4, 5, 7, 8, 9, 11])])

    x = colT.fit_transform(x)

    # Normalize Y data
    y = to_categorical(y)

    # Split the data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # Determined parameters
    n_input = x_train.shape[1]
    n_output = y_train.shape[1]

    input_layer = layers.Input(shape=(n_input,))
    hidden_layers = layers.Dense(hidden_nodes, activation=activations.sigmoid)(input_layer)
    # hidden_layers = layers.Dropout(0.2)(hidden_layers)
    hidden_layers = layers.Dense(hidden_nodes, activation=activations.relu)(hidden_layers)
    # hidden_layers = layers.Dropout(0.5)(hidden_layers)
    hidden_layers = layers.Flatten()(hidden_layers)
    output_layer = layers.Dense(n_output, activation=activations.softmax)(hidden_layers)

    model = Model(input_layer, output_layer)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              verbose=2)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss: ', score[0])
    print('Accuracy: ', score[1])
