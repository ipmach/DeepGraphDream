import numpy as np
import torch

import matplotlib.pyplot as plt
from Models.model_abstract import Model

class Dreamer(torch.nn.Module):
    def __init__(self, base_model):
        super(Dreamer, self).__init__()
        self.model = base_model
        self.activation = {}
        self.model.lin.register_forward_hook(self.create_hook('final_fc'))

    def create_hook(self, name):
      def hook(m, i, o):
        # copy the output of the given layer
        self.activation[name] = o
      return hook

    def forward(self, x, nodes,target_label, random_mask=False, steps=400):
        model = self.model
        batch = torch.zeros([nodes]).type(torch.int64)
        print(batch.shape)
        sftmax = torch.nn.Softmax()
        out = sftmax(model(x.x, x.edge_index, batch))
        print("initial prediction {}".format(out))
        mask = torch.ones(x.edge_index.shape[1], requires_grad=True)*0.5
        if random_mask:
            mask += (torch.rand(x.edge_index.shape[1]) - 0.01)*0.3

        model.eval()
        print("initial weights")
        plt.bar(range(len(mask)), mask.detach().numpy())
        plt.show()

        loss_list = []
        softmax_zero = []
        softmax_one = []

        for i in range(steps):
          #print(i)
          current_softmax = sftmax(model(example.x, example.edge_index, batch, edge_weight=mask))

          # Assumming model to be a binary classifier
          softmax_zero.append(current_softmax[0, 0])
          softmax_one.append(current_softmax[0, 1])

          if mask.grad:
            mask.grad.data.zero_()

          fc_zero = self.activation['final_fc']
          loss = torch.mean((fc_zero[:, target_label]))
          loss_list.append(loss.item())
          loss.backward()
          try:
            mask = mask + (0.005 * mask.grad.data)
          except Exception as e:
            print("i {}".format(i))
            print("no grad ? {}".format(mask.grad))
          mask = torch.clip(mask, 0, 1)
          mask = torch.autograd.Variable(mask, requires_grad=True)
        print("weights end\n {}".format(mask))
        plt.plot(loss_list, label='activation to maximize')
        plt.legend()
        plt.show()
        plt.plot(softmax_zero, label='softmax_zero')
        plt.plot(softmax_one, label='softmax_one')
        plt.legend()
        plt.show()

        print("Weights after dreaming")
        plt.bar(range(len(mask)), mask.detach().numpy())
        plt.show()
        return mask

