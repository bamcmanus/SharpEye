from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

#import system stuff
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os

#import helpers
import siamese
import visualize

#prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

#setup siamese network
siamese = siamese.Siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

#to load previously trained
load = False
model_ckpt = './model.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("we found model files. Do you want to load it and continue training[yes/no]?")
    if input_var == 'yes':
        load = True

#start training
if load: saver.restore(sess,'./model')

for step in range(5000):
    batch_x1, batch_y1 = mnist.train.next_batch(128)
    batch_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2).astype('float')

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={siamese.x1: batch_x1, siamese.x2: batch_x2, siamese.y: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step%10 == 0:
        print('step %d: loss %.3f' % (step, loss_v))

    if step%1000 == 0 and step > 0:
        saver.save(sess, './model')
        embed = siamese.o1.eval({siamese.x1: mnist.test.images})
        embed.tofile('embed.txt')

x_test = mnist.test.images.reshape([-1,28,28])
y_test = mnist.test.labels
visualize.visualize(embed, x_test, y_test)
    

