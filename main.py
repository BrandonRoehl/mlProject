from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

if __name__ == '__main__':
    patients = pd.read_csv('processed.cleveland.data', dtype='object', header=None, names=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'depression',
        'exercise', 'ca', 'thal'
    ])
    for x in patients.columns:
        patients = patients[patients[x] != '?']
        patients = patients[patients[x] != '-9']

    patients = patients.astype(np.float64)
    patients = patients[patients.thal <= 1]
    print(patients.shape)
    print(patients.dtypes)

    column_length = len(patients.columns) - 1
    x = patients.columns[:column_length]
    y = patients.columns[column_length:]

    x_train, x_test, y_train, y_test = train_test_split(patients[x], patients[y], random_state=0)

    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(categories=[[0, 1],
                                                 [1, 2, 3, 4],
                                                 [0, 1, 2],
                                                 [1, 2, 3]]), [1, 2, 6, 10]),
         ("norm", Normalizer(norm='l1'), [0, 3, 4, 5, 7, 8, 9, 11])])

    x_train = colT.fit_transform(x_train)
    x_test = colT.fit_transform(x_test)

    learning_rate = .001
    training_epochs = 1000
    display_epochs = 100

    n_input = x_train.shape[1]
    n_length = x_train.shape[0]
    n_hidden = 72

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    n_output = y_train.shape[1]
    #
    # model = tf.keras.Sequential()
    # model.add(layers.Dense(n_hidden, activation='sigmoid', input_shape=(n_input,)))
    # model.add(layers.Dense(n_output, activation='sigmoid'))
    # model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss=tf.keras.losses.categorical_crossentropy,
    #               metrics=['accuracy'])
    #
    # model.summary()
    #
    # model.fit(x_train, y_train, epochs=training_epochs, batch_size=n_length, validation_data=(x_test, y_test),
    #           verbose=2)
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Loss: ', score[0])
    # print('Accuracy: ', score[1])

    weights = {
        "hidden": tf.Variable(tf.random_normal([n_input, n_hidden]), name="weight_hidden"),
        "output": tf.Variable(tf.random_normal([n_hidden, n_output]), name="weight_output")
    }

    bias = {
        "hidden": tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
        "output": tf.Variable(tf.random_normal([n_output]), name="bias_output")
    }


    def model(X, weights, bias):
        layer_1 = tf.add(tf.matmul(X, weights["hidden"]), bias["hidden"])
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
        train_result = sess.run(pred, feed_dict={X: x_train})

        # print(train_result)
        # print(np.argmax(train_result, 1))

        correct_pred_train = tf.equal(tf.argmax(train_result, 1), tf.argmax(y_train, 1))
        sess.run(correct_pred_train)
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

    # print(patients.shape)
# 1 2, 6, 10, 13
