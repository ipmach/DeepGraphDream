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

We define a solution space and a score for multiple dimensions. The solution space is [0,1]^n where n is the number of neurons we have in that layer. The score is 0 if there is no improve, 1 if there was a perfect improvement and -1 if the improvement was the worst possible.

<img src="https://github.com/ipmach/DeepGraphDream/blob/main/Documentation/score.png" alt="drawing" width="600"/>

## Example usage


``` python
# External library
from MetaHeuristics.Metaheuristics.simulated_anneling import SimmulatedAnneling
# Internal libraries
from Metaheuristic.edges2binary import Edges2binary, score
from Metaheuristic.problem_metaheuristic import Problem
from Metaheuristic.encode import NonEncode


def solver(model, graph, batch, index, x):
   m = torch.nn.Softmax(dim=1)
   return  score(old, m(model(graph.x, 
                             Edges2binary.convert_edges(x, edge_index), 
                             batch)), index)
                             
index = 1  # Select neuron we want to increase value

# Initialize Problem
problem = Problem(graph, batch, model, index, len(initial_solution), solver)

# Initialize Encode
encode = NonEncode()

# Apply metaheuristic for deepdream
solver = SimmulatedAnneling(problem, reduction='geometric')
_, iterations_SA = solver(solution=initial_solution, max_iter=2000, T=200)

````
