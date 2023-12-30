import torch


class SimulatorCollectionDataset:
    """A dataset class for the simulator collection"""

    def __init__(self, simulator_collection):
        """
        :param simulator_collection: the simulator collection
        """
        super(SimulatorCollectionDataset, self).__init__()
        self.simulator_collection = simulator_collection
        self.x = simulator_collection.getXs()
        self.t = simulator_collection.getTs()
        self.constants = simulator_collection.getConstantsList()

    def __getitem__(self, index):
        """
        :param index: the index of the data, which stands for one simulator
        :return: the input, and the target output
        """
        return self.x[index], self.t[index], self.constants[index]

    def getInputLength(self):
        """
        :return: the length of the input
        """
        return len(self.x[0])

    def getOutputLength(self):
        """
        :return: the length of the output
        """
        return len(self.constants[0])

    def __len__(self):
        """
        :return: the length of the dataset
        """
        return len(self.x)
