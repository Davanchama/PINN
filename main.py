from physics.spring.SpringSimulator import SpringSimulator
from SimulationPlotter import SimulationPlotter
from pinn.PinnPlotter import PINNPlotter
from pinn.PINN import PINN
import torch
from pinn.DNN import DNN
from pinn.SimulatorDataset import SimulatorDataset

if __name__ == '__main__':
    torch.manual_seed(123)

    simulator = SpringSimulator(1, 10, 0.1, 1, 0, 0)
    simulator.run(0.01, 10, verbose=False)
    plotter = SimulationPlotter(simulator)
    plotter.plotPosition()

    dataset = SimulatorDataset(simulator)
    # continuous split, models will be trained on the first 30% of the data and tested on the last 70%
    split = int(len(dataset) * 0.3)
    trainset, testset = torch.utils.data.Subset(dataset, range(0, split)), torch.utils.data.Subset(dataset, range(split, len(dataset)))

    dnn = DNN([2, 25, 1])
    dnn.traindnn(trainset, 50, 0.01)
    dnn.testdnn(testset, verbose=True, simulator=simulator)

    dnnplotter = PINNPlotter(dataset, dnn)
    dnnplotter.plot("DNN")

    pinn = PINN(simulator, [2, 25, 1])
    pinn.trainpinn(trainset, 50, 1, 0.01)
    pinn.testpinn(testset, verbose=True)

    pinnplotter = PINNPlotter(dataset, pinn)
    pinnplotter.plot("PINN")



