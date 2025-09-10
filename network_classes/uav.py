class Uav:
    def __init__(self, h, h_bar, theta, c, phi_x, phi_y, phi_x_bar, phi_y_bar):
        self.h = h  # h
        self.h_bar = h_bar  # \bar{h}
        self.theta = theta
        self.c = c  # speed of the UAV
        self.x = phi_x  # horizontal x of the UAV
        self.y = phi_y  # horizontal y of the UAV
        self.x_bar = phi_x_bar  # horizontal x of the UAV at the previous time step
        self.y_bar = phi_y_bar  # horizontal y of the UAV at the previous time step

