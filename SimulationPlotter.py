from matplotlib import pyplot as plt


class SimulationPlotter:
    # can plot positions, velocities, accelerations, etc. of one simulator
    def __init__(self, simulator):
        self.simulator = simulator
        self.timeList = simulator.timeList
        self.positionList = simulator.positionList

    def plotPosition(self):
        plt.plot(self.timeList, self.positionList)
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")
        plt.show()
