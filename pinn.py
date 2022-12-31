import tensorflow as tf
from gradlayer import GradientLayer


class Pinn:
    """Builds a physics-informed neural network"""

    def __init__(self, network, c=1.0):
        self.network = network
        self.grads = GradientLayer()
        self.grads.network = self.network
        self.c = c
        self.pinn = self._build()

    def _build(self):
        """
        Inputs = [(t, x), (t0, x), (t, x_bnd)]
        Outputs: [residual = iu_t + u_xx, u0, u_t0, u_bnd]
        """

        tx = tf.keras.layers.Input(shape=(2,))
        # t = t0, x
        tx0 = tf.keras.layers.Input(shape=(2,))
        # t, x = (x1, x2)
        tx_bnd = tf.keras.layers.Input(shape=(2,))
        inputs = [tx, tx0, tx_bnd]

        _, psi_t, psi_xx = self.grads(tx)
        u_t, v_t, = psi_t[0], psi_t[1]
        u_xx, v_xx = psi_xx[0], psi_xx[1]

        residual_u = u_t + self.c*v_xx
        residual_v = v_t - self.c*u_xx

        # This part is hardcoded since I'm only going to use real boundary and initial conditions
        psi0, _, _ = self.grads(tx0)
        psi0 = psi0[0]

        psi_bnd, _, _ = self.grads(tx_bnd)
        psi_bnd = psi_bnd[0]

        outputs = [residual_u, residual_v, psi0, psi_bnd]

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    from network import Network
    import numpy as np

    net = Network.build(hid_neurons=[4])
    pinn = Pinn(net).pinn

    x = np.random.rand(5, 2)
    x0 = np.random.rand(5, 2)
    x_bnd = np.random.rand(5, 2)

    def u0fun(tx, a=np.pi):
        t, x = tx[..., 0, None], tx[..., 1, None]
        return np.sin(a * x)


    res = np.zeros((5, 1))
    u0 = u0fun(x0)
    u_b = np.zeros((5, 1))

    outputs = [res, res, u0, u_b]

    x_train = [tf.constant(xx, dtype='float32') for xx in [x, x0, x_bnd]]
    y_train = [tf.constant(yy, dtype='float32') for yy in outputs]
    pinn.compile(optimizer='adam', loss='mse')

    pinn.fit(x_train, y_train, epochs=5, batch_size=5)
