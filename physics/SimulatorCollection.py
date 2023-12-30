import torch
class SimulatorCollection:
    """A collection of simulators. This class will be used for the identification dataset used to train a PINN
    on identifying the key constants in the physics simulator."""

    def __init__(self, constants_dict, simulator_class, dt=0.01, t_max=10.0):
        self.simulator_class = simulator_class
        self.constants_dict = constants_dict
        self.simulators = []
        self.constructSimulators()
        self.dt = dt
        self.t_max = t_max
        self.runSimulations()

    def constructSimulators(self):
        """Construct simulators with different constants"""
        for constants in self.constants_dict.values():
            constants = tuple(constants)
            self.simulators.append(self.simulator_class(*constants))


    def runSimulations(self):
        """Run all the simulations"""
        for simulator in self.simulators:
            simulator.run(self.dt, self.t_max)

    def getSimulator(self, index):
        """Get a simulator from the collection"""
        return self.simulators[index]

    def getConstants(self, index):
        """Get the constants of a simulator from the collection"""
        return self.constants_dict[index]

    def getConstantsList(self):
        """Get the constants of a simulator from the collection"""
        return list(self.constants_dict.values())

    def getX(self, index):
        """Get one input for the PINN. This is the complete data of X (and T) to compute the physics constants,
        which is the task of the PINN."""
        return self.simulators[index].getX()

    def getT(self, index):
        """Get one input for the PINN. This is the complete data of X (and T) to compute the physics constants,
        which is the task of the PINN."""
        return self.simulators[index].getT()

    def getXs(self):
        """Get all inputs for the PINN"""
        return torch.cat([simulator.getX() for simulator in self.simulators])

    def getTs(self):
        """Get all inputs for the PINN"""
        return torch.cat([simulator.getT() for simulator in self.simulators])
