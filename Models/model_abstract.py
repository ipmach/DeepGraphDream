import matplotlib.pyplot as plt
from abc import abstractmethod
from IPython import display
import torch


class Model(torch.nn.Module):
    """
    Abstract class inspired from:
        https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
    """

    @abstractmethod
    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Apply forward in the model
        :param x: features matrix
        :param edge_index: adjencency matrix
        :param batch: batch size
        :param  edge_weight: weights of the edges
        :return: output model
        """
        pass

    def save_model(self, path):
        """
        Save model
        :param path: path where to save it
        :return: None
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Load model
        :param path: path where to load it
        :return: None
        """
        self.load_state_dict(torch.load(path))

    def train_step(self, train_loader, use_weights=False):
        """
        Train step (redefine if model work different)
        :param train_loader: train loader
        :param use_weights: use edge weights
        :return: None
        """
        self.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            if use_weights:
                out = self(data.x, data.edge_index, data.batch, data.edge_weights)
            else:
                out = self(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def test_step(self, test_loader, use_weights=False):
        """
        Test step (redefine if model work different)
        :param test_loader: test loader
        :param use_weights: use edge weights
        :return: accuracy
        """
        self.eval()
        correct = 0
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            if use_weights:
                out = self(data.x, data.edge_index, data.batch, data.edge_weights)
            else:
                out = self(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.

    def do_train(self, num_epochs, train_loader, test_loader,
                 learning_rate=0.01, visualize=True, pause_plt=0.4,
                 use_weights=False):
        """
        Do training
        :param num_epochs: number of epochs to train
        :param train_loader: train loader
        :param test_loader: test loader
        :param learning_rate: learning rate
        :param visualize: visualize plot each epoch
        :param pause_plt: time to wait from visualization (train is slower)
        :param use_weights: use edge weights
        :return: train_acc, test_acc
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        train_acc = []
        test_acc = []
        for epoch in range(num_epochs):
            self.train_step(train_loader)
            train_acc.append(self.test_step(train_loader,
                             use_weights=use_weights))
            test_acc.append(self.test_step(test_loader,
                            use_weights=use_weights))
            if visualize:
                display.clear_output()
                plt.figure(1)
                plt.title("Total Accuracy")
                plt.plot(train_acc, label="train")
                plt.plot(test_acc, label="test")
                plt.legend()
                plt.show()
                plt.pause(pause_plt)

        return train_acc, test_acc
