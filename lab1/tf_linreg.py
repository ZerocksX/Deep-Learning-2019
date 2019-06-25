import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## 1. definicija računskog grafa
# podatci i parametri
X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)
batch_size = tf.cast(tf.shape(X)[0], dtype=tf.float32)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y - Y_) ** 2

dL_da = tf.reduce_sum(2 * (Y - Y_) * X, 0) / batch_size
dL_db = tf.reduce_sum(2 * (Y - Y_), 0) / batch_size

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = trainer.compute_gradients(loss, [a, b])
capped_grads_and_vars = [(gv[0] / batch_size, gv[1]) for gv in grads_and_vars]
train_op = trainer.apply_gradients(capped_grads_and_vars)
printed = tf.Print(capped_grads_and_vars, [capped_grads_and_vars, grads_and_vars, dL_da, dL_db], 'Message: ')
# train_op = trainer.minimize(loss)

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())

N = 10000

## 3. učenje
# neka igre počnu!
for i in range(100):
    val_loss, _, val_a, val_b, _, dL_da_val, dL_db_val = sess.run([loss, train_op, a, b, printed, dL_da, dL_db],
                                                                  feed_dict={X: np.linspace(1, 2, N),
                                                                             Y_: np.linspace(1, 2, N) * 2 + 1})
    print(i, np.average(val_loss), val_a, val_b)
