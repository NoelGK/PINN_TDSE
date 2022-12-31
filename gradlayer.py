import tensorflow as tf


class GradientLayer(tf.keras.layers.Layer):
    """Custom layer to perform derivatives of the wave function"""

    def __int__(self, network, **kwargs):
        self.network = network

    def call(self, tx):
        with tf.GradientTape(persistent=True) as g:
            g.watch(tx)

            with tf.GradientTape(persistent=True) as gg:
                gg.watch(tx)
                u, v = self.network(tx)

            jacob_u = gg.batch_jacobian(u, tx)
            u_t = jacob_u[..., 0]
            jacob_v = gg.batch_jacobian(v, tx)
            v_t = jacob_v[..., 0]

        hessian_u = g.batch_jacobian(jacob_u, tx)
        u_xx = hessian_u[..., 1, 1]

        hessian_v = g.batch_jacobian(jacob_v, tx)
        v_xx = hessian_v[..., 1, 1]

        return (u, v), (u_t, v_t), (u_xx, v_xx)
