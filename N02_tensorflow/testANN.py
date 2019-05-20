import tensorflow  as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100


nb_classes =10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32,[None, nb_classes])

# Layer 1
#W1 = tf.Variable(tf.random_normal([784,256]))
W1 = tf.get_variable("W1", shape=[784, 512],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# Layer 2
#W2 = tf.Variable(tf.random_normal([256,256]))
W2 = tf.get_variable("W2", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

# Layer 3
#W3 = tf.Variable(tf.random_normal([256,256]))
W3 = tf.get_variable("W3", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)

# Layer 4
#W4 = tf.Variable(tf.random_normal([256,256]))
W4 = tf.get_variable("W4", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)

# Layer 5
#W5 = tf.Variable(tf.random_normal([256,nb_classes]))
W5 = tf.get_variable("W5", shape=[512, nb_classes],initializer=tf.contrib.layers.xavier_initializer()) # Xavier_Initializer
b5 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L4,W5)+b5

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(training_epochs):
    avg_loss=0
    batch = int(mnist.train.num_examples / batch_size)

    for step in range(batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}

        c, _ = sess.run([loss,optimizer],feed_dict=feed_dict )
        avg_loss += c / batch
    print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_loss))
print('Learning finished')


# Test model and check acc
correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels}))

