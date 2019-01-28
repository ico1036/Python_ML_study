import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hyper parameters
learning_rate = 0.001
training_epoch = 15
batch_size = 15

# Place holders
X = tf.placeholder(tf.float32, [None,784])
X_img = tf.reshape(X,[-1,28,28,1])  # N numner of images with 28x28 pixel and 1 color
Y = tf.placeholder(tf.float32,[None,10])

### --NetWork 

# ------Layer1 input shape=(?,28,28,1)

## Conv L1
# Filter type: 3x3, Filter #: 32 
# Stride = 1, padding= ok
# Conv -> (?,28,28,32)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)

## Pool L1
# Pool type: max pooling, Kernnl type: 2x2
#stride = 2, padding= ok
# Pool-> (?, 14, 14, 32)
# 2 stride make 28 -> 14
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# ------Layer2 input shape=(?,14,14,32)

## Conv L2
# Filter type: 3x3, Filter #: 64 
# Stride = 1, padding= ok
# Conv -> (?,14,14,64)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)

## Pool L2
# Pool type: max pooling, Kernnl type: 2x2
#stride = 2, padding= ok
# Pool-> (?, 7, 7, 64)
# 2 stride make 28 -> 14
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1,7*7*64])


# ------FC 7X7X64 inputs --> 10 outputs
W3 = tf.get_variable("W3", shape=[7*7*64,10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat,W3)+b

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
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_loss += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_loss))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))




