from utils import load_dataset, encode
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

trainData, trainLabels = load_dataset("dataset/mnist_train.csv")
testData, testLabels = load_dataset("dataset/mnist_test.csv")

trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainLabels = encode(trainLabels)
testLabels = encode(testLabels)


X = tf.placeholder(tf.float32, shape=[None, 784], name="Inputs")
Y = tf.placeholder(tf.float32, shape=[None, 10], name="Targets")

keep_prob_1 = tf.placeholder(tf.float32, name="keepProb_1")
keep_prob_2 = tf.placeholder(tf.float32, name="keepProb_2")

def weight_bias(shape):
    with tf.name_scope("Weights_Biases"):
        with tf.name_scope("Weight"):
            w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            tf.summary.histogram("weights", w)
        with tf.name_scope("Bias"):
            b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
            tf.summary.histogram("biases", b)
    return w, b

def conv_relu_pool(input, filter_shape=(5,5,64,64)):
    weight, bias = weight_bias(filter_shape)
    with tf.name_scope("Convolution"):
        conv = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding="VALID") + bias
    with tf.name_scope("Relu"):
        relu = tf.nn.relu(conv)
    with tf.name_scope("Maxpooling"):
        pool = tf.nn.max_pool(relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    return pool

X_image = tf.reshape(X, shape=[-1, 28, 28, 1])

with tf.name_scope("ConvRelPool_1"):
    conv_1 = conv_relu_pool(X_image, filter_shape=[3,3,1,32])
with tf.name_scope("ConvRelPool_2"):
    conv_2 = conv_relu_pool(conv_1, filter_shape=[3,3,32,32])

#conv_2 = tf.nn.dropout(conv_2, keep_prob=keep_prob_1)

conv_2 = tf.reshape(conv_2, (-1, 5*5*32))

W_fc1, b_fc1 = weight_bias([5*5*32, 128])
W_fc2, b_fc2 = weight_bias([128, 10])

with tf.name_scope("Fully_Connected_1"):
    fc_1 = tf.nn.relu(tf.matmul(conv_2, W_fc1) + b_fc1)

    with tf.name_scope("Dropout_fc1"):
        fc_1 = tf.nn.dropout(fc_1, keep_prob=keep_prob_2)

with tf.name_scope("Fully_Connected_2"):
    fc_2 = tf.matmul(fc_1, W_fc2) + b_fc2
with tf.name_scope("Predictions"):
    pred = tf.nn.softmax(fc_2)

with tf.name_scope("Cost"):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=Y)
    cost_reduced = tf.reduce_mean(cost)
    tf.summary.scalar("cost", cost_reduced)

with tf.name_scope("Objective_Function"):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

with tf.name_scope("Accuracy"):
    with tf.name_scope("Correct_Prediction"):
        correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
    with tf.name_scope("Accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

merged_summaries = tf.summary.merge_all()

init = tf.initialize_all_variables()
batch_size = 128

with tf.Session() as sess:
    sess.run(init)

    train_writer = tf.summary.FileWriter("logs/train_1", graph=sess.graph)
    test_writer = tf.summary.FileWriter("logs/test_1", graph=sess.graph)

    for i in xrange(12):
        costs = []
        # MiniBatch descent
        for start, end in zip(range(0, len(trainData), batch_size), range(batch_size, len(trainData) + 1, batch_size)):
            _,train_summary = sess.run([train_op, merged_summaries],
                            feed_dict={X: trainData[start:end], Y: trainLabels[start:end], keep_prob_1: 0.5,
                                       keep_prob_2: 0.75})
            
            test_idxs = np.random.choice(len(testLabels), batch_size)
            testData_batch = testData[test_idxs]
            testLabels_batch = testLabels[test_idxs]
            test_summary,c,acc = sess.run([merged_summaries,cost,accuracy],
                       feed_dict={X: testData_batch, Y: testLabels_batch, keep_prob_1: 1.0, keep_prob_2: 1.0})

            train_writer.add_summary(train_summary,i*(start+1))
            test_writer.add_summary(test_summary, i*(start+1))

            costs.append(c)
        c = np.mean(np.array(costs))
        
        print "Epoch: {}, Cost: {}, Accuracy: {}".format(i + 1, c, acc)
    train_writer.close()
    test_writer.close()

print "[INFO] optimization finished"

