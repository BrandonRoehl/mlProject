from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical

if __name__ == '__main__':
    # Hyper parameters
    learning_rate = .001
    training_epochs = 5000
    display_epochs = 100
    hidden_nodes = 10
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

    weights = {
        "hidden": tf.Variable(tf.random_normal([n_input, hidden_nodes]), name="weight_hidden"),
        "output": tf.Variable(tf.random_normal([hidden_nodes, n_output]), name="weight_output")
    }

    bias = {
        "hidden": tf.Variable(tf.random_normal([hidden_nodes]), name="bias_hidden"),
        "output": tf.Variable(tf.random_normal([n_output]), name="bias_output")
    }


    def model(x, weights, bias):
        layer_1 = tf.add(tf.matmul(x, weights["hidden"]), bias["hidden"])
        layer_1 = tf.nn.relu(layer_1)

        layer_1 = tf.nn.dropout(layer_1,0.5)

        layer_1 = tf.add(tf.matmul(layer_1, weights["hidden2"]), bias["hidden2"])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_1 = tf.nn.dropout(layer_1,0.2)


        output_layer = tf.matmul(layer_1, weights["output"]) + bias["output"]
        return output_layer


    X = tf.placeholder(data_type, [None, n_input])
    Y = tf.placeholder(data_type, [None, n_output])
    pred = model(X, weights, bias)

    # Define loss and optimizer

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initializing global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
            if (epoch + 1) % display_epochs == 0:
                # a = 5
                print("Epoch: ", (epoch + 1), "Cost: ", c)
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
        train_accuracy = tf.reduce_mean(tf.cast(correct_pred_train, data_type))
        test_accuracy = tf.reduce_mean(tf.cast(correct_pred_test, data_type))

        print(sess.run(train_accuracy))
        print(sess.run(test_accuracy))

        print("Train Accuracy: ", train_accuracy.eval({X: x_train, Y: y_train}))
        print("Test Accuracy: ", test_accuracy.eval({X: x_test, Y: y_test}))

        """"""
        print("")
        # tvars = tf.trainable_variables()
        # tvars_vals = sess.run(tvars)
        #
        # for var, val in zip(tvars, tvars_vals):
        #     print(var.name, val)

        sess.close()
