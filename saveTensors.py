# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:00:11 2018

@author: Saad
"""
import tensorflow as tf  
import numpy as np 
import matplotlib.pyplot as plt  
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Clear the current graph in each run, to avoid variable duplication
tf.reset_default_graph()

# Create placeholders for the x and y points
X = tf.placeholder("float")  
Y = tf.placeholder("float")

# Initialize the two parameters that need to be learned
h_est = tf.Variable(0.0, name='hor_estimate')  
v_est = tf.Variable(0.0, name='ver_estimate')

# y_est holds the estimated values on y-axis
y_est = tf.square(X - h_est) + v_est

# Define a cost function as the squared distance between Y and y_est
cost = (tf.pow(Y - y_est, 2))

# The training operation for minimizing the cost function. The
# learning rate is 0.001
trainop = tf.train.GradientDescentOptimizer(0.001).minimize(cost) 

# Use some values for the horizontal and vertical shift
h = 1  
v = -2

# Generate training data with noise
x_train = np.linspace(-2,4,201)  
noise = np.random.randn(*x_train.shape) * 0.4  
y_train = (x_train - h) ** 2 + v + noise

# Visualize the data 
plt.rcParams['figure.figsize'] = (10, 6)  
plt.scatter(x_train, y_train)  
plt.xlabel('x_train')  
plt.ylabel('y_train')  

##################################################################
#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")
feed_dict ={w1:4,w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1,w2, name="w3")
w4 = tf.multiply(w3,b1,name="op_to_restore")
#################################################################

t_est = tf.Variable(10.0, name='t_estimate')
h_ = tf.placeholder("float", name = "h_")
init = tf.global_variables_initializer()

# Run a session. Go through 100 iterations to minimize the cost
def train_graph():  
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            for (x, y) in zip(x_train, y_train):

                # Feed actual data to the train operation
                sess.run(trainop, feed_dict={X: x, Y: y})
              
        print(sess.run(w4,feed_dict))
        h_ = sess.run(tf.multiply(2,4))
        v_ = sess.run(v_est)
        
        # Create a Saver object
        saver = tf.train.Saver()
       
        # Save the final model
        saver.save(sess, './model_final')
        
    return h_, v_

result = train_graph()  
print("h_est = %.2f, v_est = %.2f" % result) 