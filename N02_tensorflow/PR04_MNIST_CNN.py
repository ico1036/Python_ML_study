import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hyper parameters
learning_rate = 0.001
training_epoch = 15
batch_size = 100

# Place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# Dropout
keep_prob = tf.placeholder(tf.float32)


### --NetWork 

# ------Layer1 input shape=(?,28,28,1)

## Conv L1
# Filter type: 3x3, Filter #: 32 
# Stride = 1, padding= ok
# Conv -> (?,28,28,32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

## Pool L1
# Pool type: max pooling, Kernnl type: 2x2
#stride = 2, padding= ok
# Pool-> (?, 14, 14, 32)
# 2 stride make 28 -> 14
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# ------Layer2 input shape=(?,14,14,32)

## Conv L2
# Filter type: 3x3, Filter #: 64 
# Stride = 1, padding= ok
# Conv -> (?,14,14,64)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)

## Pool L2
# Pool type: max pooling, Kernnl type: 2x2
#stride = 2, padding= ok
# Pool-> (?, 7, 7, 64)
# 2 stride make 28 -> 14
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)



# ------Layer3 input shape=(?,7,7,64)

## Conv L3
# Filter type: 3x3, Filter #: 128 
# Stride = 1, padding= ok
# Conv -> (?,7,7,128)
# Pool-> (?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3_flat = tf.reshape(L3, [-1,4*4*128])


# ------FC1 4X4X128 inputs --> 625 outputs
W4 = tf.get_variable("W4", shape=[4*4*128,625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat,W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# ------FC1 625 inputs --> 10 outputs
W5 = tf.get_variable("W5", shape=[625,10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4,W5) + b5


### -- NetWork end

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



# Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
print('Learning started')

for epoch in range(training_epoch):
    avg_loss = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:0.7}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_loss += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_loss))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels,keep_prob:1}))




