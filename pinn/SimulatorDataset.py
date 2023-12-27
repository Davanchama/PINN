import torch


class SimulatorDataset(torch.utils.data.Dataset):
    """
    A dataset class for the simulator
    """

    def __init__(self, simulator):
        """
        :param simulator: the physics simulator
        """
        super(SimulatorDataset, self).__init__()
        self.simulator = simulator
        self.x = simulator.getX().flatten().unsqueeze(1)
        self.t = simulator.getT().flatten().unsqueeze(1)
        self.u_target = simulator.getU().flatten().unsqueeze(1)

    def __getitem__(self, index):
        """
        :param index: the index of the data
        :return: the input, and the target output
        """
        return self.x[index], self.t[index], self.u_target[index]

    def __len__(self):
        """
        :return: the length of the dataset
        """
        return len(self.x)
