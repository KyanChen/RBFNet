from random import random
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class RbfNet(nn.Module):
    def __init__(self, centers, outputs_dim=4):
        super(RbfNet, self).__init__()
        self.outputs_dim = outputs_dim
        self.num_centers = centers.size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers)/10)
        self.linear = nn.Linear(self.num_centers, self.outputs_dim, bias=True)

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        C = self.centers.view(self.num_centers, -1).repeat(n_input,1,1)
        X = batches.view(n_input, -1).unsqueeze(dim=1).repeat(1, self.num_centers, 1)
        out = torch.exp(-self.beta.mul((X-C).pow(2).sum(dim=2,keepdim=False).sqrt()))
        return out

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(radial_val)
        return class_score


def gen_data(input_dims=6, output_dim=4, n_samples=1000):
    inputs = np.random.rand(n_samples, input_dims)
    outputs = np.random.rand(n_samples, output_dim)
    outputs[:, 0] = inputs[:, 0] + 2 * inputs[:, 1]
    outputs[:, 1] = inputs[:, 2] + 1 * inputs[:, 0]
    outputs[:, 2] = inputs[:, 4] + 3 * inputs[:, 3]
    outputs[:, 3] = inputs[:, 5] + 5 * inputs[:, 2] + 5 * inputs[:, 1]
    return inputs, outputs


def train(inputs, outputs, num_center=100, val_inputs=None, val_outputs=None, n_epochs=10000):
    lr = 1e-4
    loss_fun = nn.functional.mse_loss

    inputs = torch.tensor(inputs).float()
    outputs = torch.tensor(outputs).float()

    model = RbfNet(centers=torch.rand(num_center, inputs.size(-1)))
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()
    loss_save_list = []
    for epoch in range(n_epochs):

        optimizer.zero_grad()

        pred = model(inputs)
        loss = loss_fun(pred, outputs)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss.item():.4f}')
        loss_save_list.append(loss.item())
    plt.plot(loss_save_list)
    plt.show()
    torch.save(model.state_dict(), 'model_parameters.pth')


if __name__ == '__main__':
    inputs, outputs = gen_data()
    train(inputs, outputs)
