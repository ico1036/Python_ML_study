import tensorflow as tf
import numpy as np

# read data
xy=np.loadtxt('data.csv', delimiter=',', dtype=np.float)
np.random.shuffle(xy)
x_data = xy[:,1:-1]
y_data = xy[:,[-1]]

# training data & test data
tr = int(x_data.shape[0]*0.7)
x_train = x_data[0:tr,:]
y_train = y_data[0:tr,:]
x_test = x_data[tr:,:]
y_test = y_data[tr:,:]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# HyperParameter
batch_size = 32
training_epochs=10
learning_rate = 0.001

# Place holder
X = tf.placeholder(tf.float32,[None,3])
Y = tf.placeholder(tf.float32,[None,1])

# Dropout
keep_prob = tf.placeholder(tf.float32)

# Input Layer
W1 = tf.get_variable("W1", shape=[3, 50],initializer=tf.contrib.layers.xavier_initializer()) 
b1  = tf.Variable(tf.random_normal([50]),name='bias1')
L1  =tf.nn.relu(tf.matmul(X,W1)+b1)
L1 =tf.nn.dropout(L1, keep_prob=keep_prob)

# Output Layer
W2 = tf.get_variable("W2", shape=[50, 3],initializer=tf.contrib.layers.xavier_initializer()) 
b2  = tf.Variable(tf.random_normal([3]),name='bias2')
hypothesis  =tf.nn.sigmoid(tf.matmul(L1,W2)+b2)


# loss, optimizer
loss = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Acc & Prediction
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Initialize 
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Train
print('Learning started')

for epoch in range(training_epochs):
    batch_count = int(x_train.shape[0]/batch_size)
    avg_loss = 0
    for i in range(batch_count):
        batch_xs, batch_ys = x_train[i*batch_size : i*batch_size+batch_size],y_train[i*batch_size : i*batch_size+batch_size]
        feed_dict = feed_dict={X: batch_xs, Y: batch_ys , keep_prob: 0.7}
        c,acc,_ = sess.run([loss,accuracy,train],feed_dict=feed_dict) 
        avg_loss += c / batch_count
    print('Epoch:', '%04d' % (epoch+1), "Loss =, {:.9f}   Acc: {:.2%}".format(avg_loss,acc))


print('Learning Finished')


#Test model and check accuracy
print('Accuracy', sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob:1}))


