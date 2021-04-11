import tensorflow as tf

# w = tf.Variable(2.)

# def f(w):
#     y = w**2
#     z = 2*y + 5
#     return z

# with tf.GradientTape() as tape:
#     z = f(w)

# gradients = tape.gradient(z, [w])
# print(gradients)

W = tf.Variable(4.0)
b = tf.Variable(1.0)

def hypothesis(x):
    return W*x + b

x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())

@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

X = [1,2,3,4,5,6,7,8,9]
y = [11,22,33,44,55,66,77,88,99]

optimizer = tf.optimizers.SGD(0.01)


for i in range(301):
    with tf.GradientTape() as tape:

        y_pred = hypothesis(X)

        cost = mse_loss(y_pred, y)

    gradients = tape.gradient(cost, [W,b])

    optimizer.apply_gradients(zip(gradients, [W,b]))

    if i % 10 == 0:
        print("epoch : {:3} | W : {:5.4f} | b : {:5.4f} | cost : {:5.6f}".format(i, W.numpy(), b.numpy(), cost))