from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

if __name__ == '__main__':
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
    patients = patients.astype(np.float64)
    # Only train on patients with have or don't
    patients = patients[patients.thal <= 1]

    print(patients.shape)
    # => (212 , 14)
    print(patients.dtypes)
    # => float64

    # Split into test and train data
    y = patients.take([13], axis=1)
    patients = patients.drop(['thal'], axis=1)

    # Normalize X data
    colT = ColumnTransformer(
        [("onehot", OneHotEncoder(categories=[[0, 1],
                                              [1, 2, 3, 4],
                                              [0, 1, 2],
                                              [1, 2, 3]]), [1, 2, 6, 10]),
         ("norm", Normalizer(norm='l1'), [0, 3, 4, 5, 7, 8, 9, 11])])

    patients = colT.fit_transform(patients)

    # Normalize Y data
    y = to_categorical(y)

    # Split the data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(patients, y, random_state=0)

    # Hyper parameters
    learning_rate = .001
    training_epochs = 2000
    display_epochs = 100
    n_hidden = 72

    # Determined parameters
    n_input = x_train.shape[1]
    n_length = x_train.shape[0]
    n_output = y_train.shape[1]

    weights = {
        "hidden": tf.Variable(tf.random_normal([n_input, n_hidden]), name="weight_hidden"),
        "output": tf.Variable(tf.random_normal([n_hidden, n_output]), name="weight_output")
    }

    bias = {
        "hidden": tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
        "output": tf.Variable(tf.random_normal([n_output]), name="bias_output")
    }


    def model(x, weights, bias):
        layer_1 = tf.add(tf.matmul(x, weights["hidden"]), bias["hidden"])
        layer_1 = tf.nn.relu(layer_1)

        output_layer = tf.matmul(layer_1, weights["output"]) + bias["output"]
        return output_layer


    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    pred = model(X, weights, bias)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initializing global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
            if (epoch + 1) % display_epochs == 0:
                # a = 5
                print("Epoch: ", (epoch + 1), " Cost: ", c)
        print("Optimization finished!")

        # Get train and test results and compare them to the expected values

        # Train results
        train_result = sess.run(pred, feed_dict={X: x_train})
        correct_pred_train = tf.equal(tf.argmax(train_result, 1), tf.argmax(y_train, 1))
        sess.run(correct_pred_train)
        # print(train_result)
        # print(np.argmax(train_result, 1))

        # Test results
        test_result = sess.run(pred, feed_dict={X: x_test})
        correct_pred_test = tf.equal(tf.argmax(test_result, 1), tf.argmax(y_test, 1))
        sess.run(correct_pred_test)
        # print(test_result)
        # print(np.argmax(test_result, 1))

        # Calculate the accuracy for each dataset
        train_accuracy = tf.reduce_mean(tf.cast(correct_pred_train, "float"))
        test_accuracy = tf.reduce_mean(tf.cast(correct_pred_test, "float"))

        print(sess.run(train_accuracy))
        print(sess.run(test_accuracy))

        print("Train Accuracy: ", train_accuracy.eval({X: x_train, Y: y_train}))
        print("Test Accuracy: ", test_accuracy.eval({X: x_test, Y: y_test}))

        """"""
        print("")
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)
        #
        # for var, val in zip(tvars, tvars_vals):
        #     print(var.name, val)

        sess.close()
