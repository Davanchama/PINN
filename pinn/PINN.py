import torch
import torch.nn as nn
import numpy as np
from pinn.DNN import DNN


class PINN:
    """
    A physics-informed neural network class using the DNN class
    """

    def __init__(self, simulator, layers):
        """
        :param simulator: the physics simulator
        :param layers: a list of layer sizes, e.g. [2, 20, 20, 20, 1]
        """
        super(PINN, self).__init__()
        self.simulator = simulator
        self.dnn = DNN(layers)
        self.loss_func = nn.MSELoss()

    def forward(self, x, t):
        """
        :param x: the input
        :return: the output of the network
        """
        u = self.dnn(x, t)
        return u

    def loss(self, x, t, u, u_target):
        """
        :param u_target: the target output
        :param x: the input
        :param u: the output of the network
        :return: the loss of the network
        """
        derivatives = self.dnn.getDerivative(x, t)
        u_t = derivatives[0]
        u_x = derivatives[1]
        u_xx = derivatives[2]
        # f is the residual of the PDE, which should be 0. the simulator calculates the residual
        f = self.simulator.calculatePDE(u, u_x, u_xx)
        # the loss is the mean squared error of the pred and the actual plus the mean squared error of the residual
        return self.loss_func(u, u_target) + 0.02*self.loss_func(f, torch.zeros_like(f))

    def trainpinn(self, dataset, epochs, batch_size, learning_rate, optimizer=torch.optim.Adam, verbose=False):
        """
        :param dataset: the dataset to train on
        :param epochs: the number of epochs to train for
        :param batch_size: the batch size
        :param learning_rate: the learning rate
        :param optimizer: the optimizer to use
        """
        optimizer = optimizer(self.dnn.parameters(), lr=learning_rate)
        self.dnn.train()
        for epoch in range(epochs):
            # shuffle the dataset
            index_list = np.arange(len(dataset))
            np.random.shuffle(index_list)
            for k in index_list:
                x = dataset[k:k + batch_size][0]
                t = dataset[k:k + batch_size][1]
                u_target = dataset[k:k + batch_size][2]
                # bring tensor to shape 1,n instead of n,1
                x = x.transpose(0, 1)
                t = t.transpose(0, 1)
                u_target = u_target.transpose(0, 1)
                u = self.forward(x, t)
                loss = self.loss(x, t, u, u_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if verbose:
                print("Epoch {0}: {1}".format(epoch, loss.item()))

    def testpinn(self, dataset, batch=1, verbose=False):
        """
        Does not count residual loss
        :param dataset: the dataset to test on
        :return: the loss of the network
        """
        losses = []
        pdes = []
        self.dnn.eval()
        for k in range(0, len(dataset), batch):
            x = dataset[k:k + batch][0]
            t = dataset[k:k + batch][1]
            u_target = dataset[k:k + batch][2]
            # bring tensor to shape 1,n instead of n,1
            x = x.transpose(0, 1)
            t = t.transpose(0, 1)
            u_target = u_target.transpose(0, 1)
            u = self.forward(x, t)
            loss = torch.nn.MSELoss()(u, u_target)  # does not count residual loss
            if verbose:
                losses.append(loss.item())
                pde = self.simulator.calculatePDE(u, self.dnn.getDerivative(x, t)[1], self.dnn.getDerivative(x, t)[2])
                pdes.append(pde.item())
        if verbose:
            print("Loss: {0}".format(np.mean(losses)))
            print("PDE: {0}".format(np.mean(pdes)))
