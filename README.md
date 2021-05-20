# DeepGraphDream

Framework used to apply deep dream to different graphs. Two main methods are propose:
- **Non Gradient approach**: working without ascent descend.
- **Gradient approach**: working with ascent descend.

## Code structure

Structure of the code.

### Dataset

We implemented TUD graph datasets and our synthetic dataset. Abstract class to obtain datasets:

``` python
  def __getitem__(self, item): pass  # Get graph from dataset
  def __len__(self): pass  # Return size of the dataset
  def shuffle_data(self, seed=12345): pass  # Shuffle dataset
  def get_loader(self, train_percentage=0.75, batch_size=64): pass  # Pass data to the loaders
  def view_graph_index(self, index, with_labels=True, figsize=(6,6)): pass  # Visualize graph from index
  def view_graph(data, with_labels=True, figsize=(6,6)): pass  # Visualize graph
````

#### Example usage

Example of how to import a dataset in the framework.

``` python
from Datasets.Syntheticdata import SyntheticData
from Datasets.TUDdata import TUDData

dataset = SyntheticData(5000)  # Load Synthetic dataset
#dataset = TUDData('AIDS')  # Load TUD dataset

dataset.shuffle_data()
train_loader, test_loader = dataset.get_loader()
````

### Models

Models abtract class

``` python
 @abstractmethod
 def forward(self, x, edge_index, batch, edge_weight=None): pass  # Apply forward in the model
 def save_model(self, path): pass  # Save model
 def load_model(self, path): pass  # Load model
 def train_step(self, train_loader): pass # Train step
 def test_step(self, test_loader): pass  # Test step
 def do_train(self, num_epochs, train_loader, test_loader,
                 learning_rate=0.01, visualize=True, pause_plt=0.4): pass  # Do training
````

#### Example usage

``` python
from Models.GCN import GCN

model = GCN(hidden_channels=64, dataset=dataset)  # Initialize model 
train_acc, test_acc = model.do_train(100, train_loader, test_loader)  # Train model
````

## Non gradient approach

We use different metaheuristics for this approach working with the edge binary representation.

<img src="https://github.com/ipmach/DeepGraphDream/blob/main/Documentation/metaheuristic.png" alt="drawing" width="600"/>

### Solution space and score definition

We define a solution space and a score.

<img src="https://github.com/ipmach/DeepGraphDream/blob/main/Documentation/score.png" alt="drawing" width="600"/>
