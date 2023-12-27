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

    def getPositionList(self):
        return self.positionList

    def getVelocityList(self):
        return self.velocityList

    def getAccelerationList(self):
        return self.accelerationList

    def getTimeList(self):
        return self.timeList

