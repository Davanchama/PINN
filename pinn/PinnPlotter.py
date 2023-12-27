import matplotlib.pyplot as plt


class PINNPlotter:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.x, self.t, self.u, self.u_target = self.getPredictions()

    def getPredictions(self):
        x = []
        t = []
        u = []
        u_target = []
        for i in range(len(self.dataset)):
            x_i = self.dataset[i][0]
            t_i = self.dataset[i][1]
            u_i = self.model.forward(x_i, t_i)
            u_target_i = self.dataset[i][2]
            x.append(x_i)
            t.append(t_i)
            u.append(u_i)
            u_target.append(u_target_i)
        return x, t, u, u_target

    def plot(self, title):
        """ plot the real and predicted data overlapping in one plot """
        # first, tensors must be converted to numpy arrays
        x = [x_i.numpy() for x_i in self.x]
        t = [t_i.numpy() for t_i in self.t]
        u = [u_i.detach().numpy() for u_i in self.u]
        u_target = [u_target_i.numpy() for u_target_i in self.u_target]
        # plot
        plt.plot(t, u_target, label="target")
        plt.plot(t, u, label="predicted")
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title(title)
        plt.legend()
        plt.show()
