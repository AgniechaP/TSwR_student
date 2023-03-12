import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        # q - wyjście z obiektu
        # qr - wartość zadana
        q1, q2, q1_dot, q2_dot = x

        q_t = x[:2]
        q_t_dot = x[2:]

        K_d = [[250, 0], [0, 30]]
        K_p = [[250, 0], [0, 30]]

        v = q_r_ddot + K_d @ (q_r_dot - q_t_dot) + K_p @ (q_r - q_t)

        tau = self.model.M(x) @ v[:, np.newaxis] + self.model.C(x) @ q_r_dot[:, np.newaxis]

        return tau
