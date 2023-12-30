# the abstract class for every simulator
class Simulator:
    def __init__(self, initialPosition, initialVelocity, initialAcceleration, gravity=9.81):
        self.initialPosition = initialPosition
        self.initialVelocity = initialVelocity
        self.initialAcceleration = initialAcceleration
        self.position = initialPosition
        self.velocity = initialVelocity
        self.acceleration = initialAcceleration
        self.time = 0
        self.timeList = []
        self.positionList = []
        self.velocityList = []
        self.accelerationList = []
        self.gravity = gravity

    def update(self, dt):
        pass

    def calculatePDE(self, *args):
        pass

    def appendStateToLists(self):
        # to be called at the end of update
        self.timeList.append(self.time)
        self.positionList.append(self.position)
        self.velocityList.append(self.velocity)
        self.accelerationList.append(self.acceleration)

    def run(self, dt, time, verbose=False):
        if verbose:
            print(self)
        for i in range(int(time / dt)):
            self.update(dt)
            if verbose:
                print(self)

    def getPosition(self):
        return self.position

    def getVelocity(self):
        return self.velocity

    def getAcceleration(self):
        return self.acceleration

    def getTime(self):
        return self.time

    def getInitialPosition(self):
        return self.initialPosition

    def getInitialVelocity(self):
        return self.initialVelocity

    def getInitialAcceleration(self):
        return self.initialAcceleration

    def getX(self):
        pass

    def getV(self):
        pass

    def getA(self):
        pass

