from utils import load_dataset, encode
import tensorflow as tf
import numpy as np

#load the training and testing dataset
trainData, trainLabels = load_dataset("dataset/mnist_train.csv")
testData, testLabels = load_dataset("dataset/mnist_test.csv")

#convert to floats and one-hot encode the labels
trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainLabels = encode(trainLabels)
testLabels = encode(testLabels)

#placeholders for inputs and targets
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

#dropout keep probabilities placeholders
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)

#creates and returns random weight and bias with given shape
def weight_bias(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    return w, b

#passes input through CONVOLUTION --> RELU --> MAXPOOL
def conv_relu_pool(input, filter_shape=(5,5,64,64)):
    weight, bias = weight_bias(filter_shape)
    conv = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding="VALID") + bias
    relu = tf.nn.relu(conv)
    pool = tf.nn.max_pool(relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    return pool

#reshape the input to 28x28
X_image = tf.reshape(X, shape=[-1, 28, 28, 1])

#pass through two CONV-->RELU-->POOL layers
conv_1 = conv_relu_pool(X_image, filter_shape=[3,3,1,32])
conv_2 = conv_relu_pool(conv_1, filter_shape=[3,3,32,32])

#conv_2 = tf.nn.dropout(conv_2, keep_prob=keep_prob_1)

#flatten the previous output from convolution layer
conv_2 = tf.reshape(conv_2, (-1, 5*5*32))

#weights, biases for two affine layers
W_fc1, b_fc1 = weight_bias([5*5*32, 128])
W_fc2, b_fc2 = weight_bias([128, 10])

#fully connected layer followed by dropout
fc_1 = tf.nn.relu(tf.matmul(conv_2, W_fc1) + b_fc1)
fc_1 = tf.nn.dropout(fc_1, keep_prob=keep_prob_2)

#final output layer
fc_2 = tf.matmul(fc_1, W_fc2) + b_fc2
pred = tf.nn.softmax(fc_2)

#cost and training operations
cost = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=Y)
train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)


correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initialize all variables
init = tf.initialize_all_variables()

batch_size = 128

with tf.Session() as sess:
    sess.run(init)
    for i in xrange(12):
        costs = []
        # MiniBatch descent
        for start, end in zip(range(0, len(trainData), batch_size), range(batch_size, len(trainData) + 1, batch_size)):
            _, c = sess.run([train_op, cost],
                            feed_dict={X: trainData[start:end], Y: trainLabels[start:end], keep_prob_1: 0.5,
                                       keep_prob_2: 0.75})

            #get random test indices and fetch random test data of certain batch_size
            test_idxs = np.random.choice(len(testLabels), batch_size)
            testData_batch = testData[test_idxs]
            testLabels_batch = testLabels[test_idxs]
            c,acc = sess.run([cost,accuracy],
                       feed_dict={X: testData_batch, Y: testLabels_batch, keep_prob_1: 1.0, keep_prob_2: 1.0})

            costs.append(c)
        c = np.mean(np.array(costs))
        print "Epoch: {}, Cost: {}, Accuracy: {}".format(i + 1, c, acc)

print "[INFO] optimization finished"