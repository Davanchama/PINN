import math
import torch
from physics.Simulator import Simulator


class SpringSimulator(Simulator):
    """
    A class that simulates a spring with a given mass and spring constant.

    """

    def __init__(self, mass, springConstant, damping, initialPosition, initialVelocity, initialAcceleration,
                 gravity=9.81):
        super().__init__(initialPosition, initialVelocity, initialAcceleration, gravity)
        self.springConstant = springConstant
        self.mass = mass
        self.gravity = 0.0
        self.damping = damping

    def update(self, dt):
        # update the position, velocity, acceleration, and time
        self.time += dt
        m = self.mass
        k = self.springConstant
        x = self.initialPosition
        v = self.initialVelocity
        t = self.time
        c = self.damping
        # calculate omega
        omega = math.sqrt(k / m - (c**2 / (4 * m**2)))
        # calculate the new position
        self.position = math.exp(-c * t / (2 * m)) * (x * math.cos(omega * t) + (v / omega) * math.sin(omega * t))
        # calculate the new velocity
        self.velocity = math.exp(-c * t / (2 * m)) * (omega * math.cos(omega * t) * (
                2*m*v-c*x) - math.sin(omega * t) * (c * v + 2 * m * omega ** 2 * x)) / (2 * m * omega)
        # calculate the new acceleration
        self.acceleration = - (1/(4*m**2 * omega)) * math.exp(-c * t / (2 * m)) * (
                (4 * m * (m*v - c*x) * omega**2 - c**2 * v) * math.sin(omega * t)
                + omega * (4 * m**2 * omega**2 * x + 4 * m * c * v - c**2 * x) * math.cos(omega * t))
        self.appendStateToLists()

    def getAnalyticalSolution(self, x, t):
        """Returns the analytical solution of the differential equation."""
        m = self.mass
        k = self.springConstant
        c = self.damping
        x = self.initialPosition
        v = self.initialVelocity
        # calculate omega
        omega = math.sqrt(k / m - (c**2 / (4 * m**2)))
        # calculate the new position
        return math.exp(-c * t / (2 * m)) * (x * math.cos(omega * t) + (v / omega) * math.sin(omega * t))

    def calculatePDE(self, position, velocity, acceleration):
        # this is the partial differential equation that describes the spring
        # used by the PINN to verify the simulation, should always return 0
        # x'' + (k/m)x + g = 0
        return acceleration + (self.springConstant / self.mass) * position + self.gravity

    def calculateOwnPDE(self):
        # just without the parameters
        return self.acceleration + (self.springConstant / self.mass) * self.position + self.gravity

    def run(self, dt, time, verbose=False):
        if verbose:
            print(self)
        for i in range(int(time / dt)):
            self.update(dt)
            if verbose:
                print(self)

    def getSpringConstant(self):
        return self.springConstant

    def getMass(self):
        return self.mass

    def getDamping(self):
        return self.damping

    def getX(self):
        """Returns all calculated states of the spring as a pytorch tensor (position, velocity, acceleration),
         excluding the last state."""
        return torch.tensor([self.positionList[:-1]])

    def getT(self):
        """Returns the time list of the spring as a pytorch tensor."""
        return torch.tensor([self.timeList[:-1]])

    def getXNext(self):
        """Returns all calculated next states of the spring as a pytorch tensor (position, velocity, acceleration)."""
        return torch.tensor([self.positionList[1:]])

    def __str__(self):
        return "SpringSimulator: PDE = " + str(self.calculateOwnPDE()) + ", position = " + str(self.position) + ", velocity = " + str(
            self.velocity) + ", acceleration = " + str(self.acceleration) + ", springConstant = " + str(
            self.springConstant) + ", mass = " + str(self.mass) + ", damping = " + str(self.damping) + ", time = " + str(self.time)

    def __repr__(self):
        return self.__str__()
