import tensorflow as tf


class Network:
    @classmethod
    def build(cls, num_inputs=2, hid_neurons=None, activation='tanh'):
        if hid_neurons is None:
            hid_neurons = [16, 32, 16]

        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for u in hid_neurons:
            x = tf.keras.layers.Dense(u, activation=activation, kernel_initializer='he_normal')(x)

        real = tf.keras.layers.Dense(1, kernel_initializer='he_normal')(x)
        imag = tf.keras.layers.Dense(1, kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs, [real, imag])


if __name__ == '__main__':
    import numpy as np

    net = Network.build()
    x = tf.constant(np.random.rand(5, 2), dtype='float32')
    y = np.random.rand(5, 1)
    y = [tf.constant(asd, dtype='float32') for asd in [y, y]]

    pred = net(x)
    we = net.trainable_variables
    with tf.GradientTape() as tape:
        loss = tf.math.reduce_mean(tf.keras.losses.mse(pred, y))

    grads = tape.gradient(loss, we)
    print(grads)
