from SimulationPlotter import SimulationPlotter
from pinn.PinnPlotter import PINNPlotter
from pinn.PINN import PINN
import torch
from pinn.DNN import DNN
from pinn.SimulatorDataset import SimulatorDataset
from pinn.SimulatorCollectionDataset import SimulatorCollectionDataset
from physics.SimulatorCollection import SimulatorCollection
from physics.spring.SpringSimulator import SpringSimulator

if __name__ == '__main__':
    const_dict = {}
    const_dict_values = [[1, x, 0.1, 1, 0, 0] for x in range(1, 11)]
    const_dict_keys = ["simulator" + str(x) for x in range(1, 11)]
    for i in range(len(const_dict_keys)):
        const_dict[const_dict_keys[i]] = const_dict_values[i]
    simulator_collection = SimulatorCollection(const_dict, SpringSimulator, dt=0.01, t_max=1.0)
    dataset = SimulatorCollectionDataset(simulator_collection)

    # continuous split, models will be trained on the first 30% of the data and tested on the last 70%
    split = int(len(dataset) * 0.67)
    trainset, testset = torch.utils.data.Subset(dataset, range(0, split)), torch.utils.data.Subset(dataset, range(split,
                                                                                                                    len(dataset)))

    net_input_size = dataset.getInputLength()*2
    net_output_size = dataset.getOutputLength()

    dnn = DNN([net_input_size, net_input_size, net_output_size])
    dnn.traindnn(trainset, 50, 0.01, inference=False, verbose=True)
    dnn.testdnn(testset, verbose=True, inference=False)


def testPINNInference():
    torch.manual_seed(123)

    simulator = SpringSimulator(1, 10, 0.1, 1, 0, 0)
    simulator.run(0.01, 10, verbose=False)
    plotter = SimulationPlotter(simulator)
    plotter.plotPosition()

    dataset = SimulatorDataset(simulator)
    # continuous split, models will be trained on the first 30% of the data and tested on the last 70%
    split = int(len(dataset) * 0.3)
    trainset, testset = torch.utils.data.Subset(dataset, range(0, split)), torch.utils.data.Subset(dataset, range(split,
                                                                                                                  len(dataset)))

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



