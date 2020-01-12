# Restricted Boltzmann Machine from Scratch

<!-- > Looking down the misty path to uncertain destinations🌌🍀&nbsp;&nbsp;&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- x' <br><br><br> -->

It is implementation of Restricted Boltzmann Machine from scratch over GPU using PyTorch for tensor manipulation and GPU support

<img src = https://github.com/pushpull13/Restricted-Boltzmann-Machine/blob/master/rbm.jpg>

## Dependencies
- Torch
https://pytorch.org/
- Numpy <br>
pip install numpy
- Pandas <br>
pip install pandas
- Matplotlib <br>
pip install matplotlib

## Importing Dataset
Pandas dataframe cannot be directily converted into torch tensor <br>
So, datasets are first imported as pandas dataframe and then converted into numpy arrays which are then converted into torch tensor <br>
Here, we have inplace shuffled data
```python
dataframe = pd.read_csv("/dataset.csv", header=None)
dataset   = np.array(dataframe)
np.random.shuffle(dataset)
data      = torch.tensor(dataset[:, 1:], device="cuda")
```
## RBM
##### Parameter Initialization
<pre>
number_features : Number of features want to extracted from given dataset. It is same as number of hidden units in RBM
n_h             : Number of Hidden units
n_v             : Number of Visible units in RBM. It is same as number of inputs we have from dataset
k               : Number of Gibbs Sampling steps to perform Contrastive Divergance
epochs          : Number of epochs for number of times machine to train on same data
mini_batch_size : Size of Mini Batch over which, parameters are updated
alphs           : Learning rate of RBM
momentum        : Momentum value of RBM used to track old change in weights
weight_decay    : Rate at which, weight gets samller after every epochs
</pre>
```python
    def __init__(self, n_v=0, n_h=0, k=2, epochs=15, mini_batch_size=64, alpha=0.001, momentum=0.9, weight_decay=0.001):
        self.number_features = 0
        self.n_v             = n_v
        self.n_h             = self.number_features
        self.k               = k
        self.alpha           = alpha
        self.momentum        = momentum
        self.weight_decay    = weight_decay
        self.mini_batch_size = mini_batch_size
        self.epochs          = epochs
        self.data            = torch.randn(1, device="cuda")
```
