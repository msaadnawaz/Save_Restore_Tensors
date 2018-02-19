# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:02:05 2018

@author: B51427
"""
import tensorflow as tf  
import numpy as np 
from tensorflow.python.tools import inspect_checkpoint as chkp

tf.reset_default_graph()  
imported_meta = tf.train.import_meta_graph("model_final.meta")

with tf.Session() as sess:  
    imported_meta.restore(sess, tf.train.latest_checkpoint('./'))
    print(tf.__version__)
    chkp.print_tensors_in_checkpoint_file(file_name="./model_final", tensor_name='', all_tensors=True)#for tf v1.5#, all_tensor_names=True)
    graph = tf.get_default_graph()
    h_est2 = graph.get_tensor_by_name('hor_estimate:0')
    v_est2 = graph.get_tensor_by_name('ver_estimate:0')
#    print("h_est: %.2f, v_est: %.2f" % (h_est2, v_est2))
    print(h_est2, v_est2)
    # Access saved Variables directly
    print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved
    
    
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
    #graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    w3 = graph.get_tensor_by_name("w3:0")
    feed_dict ={w1:13.0,w2:17.0}
    
    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    print(sess.run(w3, feed_dict))
    print(sess.run(op_to_restore,feed_dict))
    #This will print 60 which is calculated 