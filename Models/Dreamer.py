import os
import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image
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

    def forward(self, x, nodes, target_label, use_edge_loss=True,
                mask=None, random_mask=False, save_evolution=False, steps=400):
        model = self.model
        batch = torch.zeros([nodes]).type(torch.int64)
        print(batch.shape)
        sftmax = torch.nn.Softmax()
        out = sftmax(model(x.x, x.edge_index, batch))
        print("initial prediction {}".format(out))
        if mask is None: #allow for user providing different weights
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
        files_bars = []
        files_lines = []

        for i in range(steps):
          #print(i)
          current_softmax = sftmax(model(x.x, x.edge_index, batch, edge_weight=mask))

          # Assumming model to be a binary classifier
          softmax_zero.append(current_softmax[0, 0])
          softmax_one.append(current_softmax[0, 1])

          if mask.grad:
            mask.grad.data.zero_()

          fc_zero = self.activation['final_fc']
          activation_loss = torch.mean((fc_zero[:, target_label]))
          loss_list.append(activation_loss.item())
          loss = activation_loss
          if use_edge_loss:
            sum_edges = torch.sum(mask)
            if target_label == 0:
                away_from_edges = -(9 - sum_edges)**2
            else:
                away_from_edges = -(8 - sum_edges)**2
            loss = loss  + away_from_edges

          loss.backward()
          try:
            mask = mask + (0.005 * mask.grad.data)
          except Exception as e:
            print("i {}".format(i))
            print("no grad ? {}".format(mask.grad))
          mask = torch.clip(mask, 0, 1)
          
          if save_evolution and (i % 2 == 0):
            filename = 'bar{}_.png'.format(i)
            files_bars.append(filename)
            plt.ylim(0, 1)
            plt.bar(range(len(mask)), mask.detach().numpy())
            plt.savefig(filename, bbox_inches='tight')
            plt.clf()

            filename = 'lines{}_.png'.format(i)
            plt.subplot(1, 2, 1)
            plt.plot(loss_list, label='activation to maximize')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(softmax_zero, label='softmax_zero')
            plt.plot(softmax_one, label='softmax_one')
            plt.legend()
            plt.savefig(filename, bbox_inches='tight')
            files_lines.append(filename)
            plt.clf()

            #visualize_new_graph(example, new_weights)
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
        self.files_bars = files_bars
        self.files_lines = files_lines
        return mask
    
    def generate_bars_gif(self, remove_files=True):
      for gif_name, filenames in [('bar.gif', self.files_bars), ('lines.gif', self.files_lines)]:
        with imageio.get_writer(gif_name, mode='I') as writer:
          for filename_ in filenames:
            image = imageio.imread(filename_)
            writer.append_data(image)
          
          # Remove files
          if remove_files:
            for filename_ in set(filenames):
              os.remove(filename_)

