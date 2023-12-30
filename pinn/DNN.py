import torch
import numpy as np
from collections import OrderedDict


def getInputs(k, dataset, inference=True):
    """A helper function for the inference function"""
    if inference:
        x = dataset[k][0]
        t = dataset[k][1]
        y = dataset[k][2]
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
    else:
        x = torch.Tensor(dataset[k][0])
        t = torch.Tensor(dataset[k][1])
        y = torch.Tensor(dataset[k][2])
    return x, t, y


class DNN(torch.nn.Module):
    """
    A deep neural network class, meant to be used standalone or for PINNs
    """

    def __init__(self, layers):
        """
        :param layers: a list of layer sizes, e.g. [2, 20, 20, 20, 1]
        """
        super(DNN, self).__init__()
        self.layers = layers
        self.activation = torch.nn.Tanh()
        self.linears = torch.nn.ModuleList()
        for k in range(len(layers) - 1):
            self.linears.append(torch.nn.Linear(layers[k], layers[k + 1]))

    def forward(self, x, t):
        """
        :param x: the input
        :param t: the time
        :return: the output of the network
        """
        try:
            u = torch.cat([x, t], dim=1)
        except:
            u = torch.cat([x, t], dim=0)
        for k in range(len(self.layers) - 2):
            u = self.activation(self.linears[k](u))
        u = self.linears[-1](u)
        return u

    def getWeights(self):
        """
        :return: a list of the weights of the network
        """
        return [self.linears[k].weight for k in range(len(self.layers) - 1)]

    def getBiases(self):
        """
        :return: a list of the biases of the network
        """
        return [self.linears[k].bias for k in range(len(self.layers) - 1)]

    def getParameters(self):
        """
        :return: a list of the parameters of the network
        """
        return self.getWeights() + self.getBiases()

    def getDerivative(self, x, t):
        """
        :param x: the input
        :param t: the time
        :return: the derivative of the network, using autograd
        """
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t, u_x, u_xx

    def traindnn(self, dataset, epochs, lr, batch=1, verbose=False, inference=True):
        """
        This trains the network using the normal loss between the output and the target and does not use PDE residuals.
        :param inference: if True, the dataset is a list of tuples, if False, the dataset is a list of lists
        :param x: the input list
        :param y: the target list
        :param epochs: the number of epochs to train for
        :param lr: the learning rate
        :param verbose: whether to print the loss
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            index_list = np.arange(len(dataset))
            np.random.shuffle(index_list)
            for k in index_list:
                x, t, y = getInputs(k, dataset, inference)
                y_pred = self.forward(x, t)
                loss = torch.nn.MSELoss()(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if verbose:
                print("Epoch {0}: {1}".format(epoch, loss.item()))

    def testdnn(self, dataset, batch=1, simulator=None, verbose=False, inference=True):
        """
        This tests the network using the normal loss between the output and the target and does not use PDE residuals.
        :param x: the input list
        :param y: the target list
        :param epochs: the number of epochs to train for
        :param lr: the learning rate
        :param verbose: whether to print the loss
        """
        losses = []
        pdes = []
        self.eval()
        for k in range(0, len(dataset), batch):
            x, t, y = getInputs(k, dataset, inference)
            y_pred = self.forward(x, t)
            loss = torch.nn.MSELoss()(y_pred, y)
            if simulator:
                pde = simulator.calculatePDE(y, self.getDerivative(x, t)[1], self.getDerivative(x, t)[2])
            if verbose:
                losses.append(loss.item())
                if simulator:
                    pdes.append(pde.item())
        if verbose:
            print("Loss: {0}".format(sum(losses) / len(losses)))
            if simulator:
                print("PDE: {0}".format(sum(pdes) / len(pdes)))
