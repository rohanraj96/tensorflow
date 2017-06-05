import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
sess=tf.Session()
# print(sess.run([node1,node2]))
node3 = tf.add(node1,node2)
# print("node 3: ",node3)
# print("actual node3 = ",sess.run(node3))
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node=a+b
# print(sess.run(adder_node,{a:3,b:9}))
# print(sess.run(adder_node,{a:[1,2],b:[2,3]}))
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model=W*x+b
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run(linear_model,{x:[1.0,2.0,3.0,4.0]}))
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(y - linear_model)
loss=tf.reduce_sum(squared_deltas)
# print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in xrange(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([W,b]))
