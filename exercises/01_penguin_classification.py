#!/usr/bin/env python
# coding: utf-8

# # Exercise 1: Classifying penguin species with PyTorch
# 
# <img src="https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png" width="750" />
# 
# 
# Artwork by @allison_horst
# 
# In this exercise, we will use the python package [``palmerpenguins``](https://github.com/mcnakhaee/palmerpenguins) to supply a toy dataset containing various features and measurements of penguins.
# 
# We have already created a PyTorch dataset which yields data for each of the penguins, but first we should examine the dataset and see what it contains.

# ### Task 1: look at the data
# In the following code block, we import the ``load_penguins`` function from the ``palmerpenguins`` package.
# 
# - Call this function, which returns a single object, and assign it to the variable ``data``.
#   - Print ``data`` and recognise that ``load_penguins`` has returned a ``pandas.DataFrame``.
# - Consider which features it might make sense to use in order to classify the species of the penguins.
#   - You can print the column titles using ``pd.DataFrame.keys()``
#   - You can also obtain useful information using ``pd.DataFrame.Series.describe()``

# In[ ]:


from palmerpenguins import load_penguins


# Let's now discuss the features we will use to classify the penguins' species, and populate the following list together:
# - ...
# - ...
# - ...

# ### Task 2: creating a ``torch.utils.data.Dataset``
# 
# All PyTorch dataset objects are subclasses of the ``torch.utils.data.Dataset`` class. To make a custom dataset, create a class which inherits from the ``Dataset`` class, implement some methods (the Python magic (or dunder) methods ``__len__`` and ``__getitem__``) and supply some data.
# 
# Spoiler alert: we've done this for you already in ``src/ml_workshop/_penguins.py``.
# 
# - Open the file ``src/ml_workshop/_penguins.py``.
# - Let's examine, and discuss, each of the methods together.
#   - ``__len__``
#     - What does the ``__len__`` method do?
#     - ...
#   - ``__getitem__``
#     - What does the ``__getitem__`` method do?
#     - ...
# - Review and discuss the class arguments.
#   - ``input_keys``— ...
#   - ``target_keys``— ...
#   - ``train``— ...
#   - ``x_tfms``— ...
#   - ``y_tfms``— ...

# ### Task 3: Obtaining training and validation datasets
# 
# - Instantiate the penguin dataloader.
#   - Make sure you supply the correct column titles for the features and the targets.
# - Iterate over the dataset
#     - Hint:
#         ```python
#         for features, targets in dataset:
#             # print the features and targets here
#         ```

# In[ ]:


from ml_workshop import PenguinDataset

data_set = PenguinDataset(
    input_keys=["bill_length_mm", "body_mass_g"],
    target_keys=["species"],
    train=True,
)

for features, target in data_set:
    pass


# - Can we give these items to a neural network, or do they need to be transformed first?
#   - Short answer: no, we can't just pass tuples of numbers or strings to a neural network.
#     - We must represent these data as ``torch.Tensor``s.

# ### Task 4: Applying transforms to the data
# 
# A common way of transforming inputs to neural networks is to apply a series of transforms using ``torchvision.transforms.Compose``. The ``Compose`` object takes a list of callable objects and applies them to the incoming data.
# 
# These transforms can be very useful for mapping between file paths and tensors of images, etc.
# 
# - Note: here we create a training and validation set.
#     - We allow the model to learn directly from the training set — i.e. we fit the function to these data.
#     - During training, we monitor the model's performance on the validation set in order to check how it's doing on unseen data. Normally, people use the validation performance to determine when to stop the training process.
# - For the validation set, we choose ten males and ten females of each species. This means the validation set is less likely to be biased by sex and species, and is potentially a more reliable measure of performance. You should always be _very_ careful when choosing metrics and splitting data.

# In[ ]:


from torchvision.transforms import Compose

# Apply the transforms we need to the PenguinDataset to get out inputs
# targets as Tensors.


# ### Task 5: Creating ``DataLoaders``—and why
# 
# - Once we have created a ``Dataset`` object, we wrap it in a ``DataLoader``.
#   - The ``DataLoader`` object allows us to put our inputs and targets in mini-batches, which makes for more efficient training.
#     - Note: rather than supplying one input-target pair to the model at a time, we supply "mini-batches" of these data at once.
#     - The number of items we supply at once is called the batch size.
#   - The ``DataLoader`` can also randomly shuffle the data each epoch (when training).
#   - It allows us to load different mini-batches in parallel, which can be very useful for larger datasets and images that can't all fit in memory at once.
# 
# 
# Note: we are going to use batch normalisation layers in our network, which don't work if the batch size is one. This can happen on the last batch, if we don't choose a batch size that evenly divides the number of items in the data set. To avoid this, we can set the ``drop_last`` argument to ``True``. The last batch, which will be of size ``len(data_set) % batch_size`` gets dropped, and the data are reshuffled. This is only relevant during the training process - validation will use population statistics.

# In[ ]:


from torch.utils.data import DataLoader

# Create training and validation DataLoaders.


# ### Task 6: Creating a neural network in PyTorch
# 
# Here we will create our neural network in PyTorch, and have a general discussion on clean and messy ways of going about it.
# 
# - First, we will create quite an ugly network to highlight how to make a neural network in PyTorch on a very basic level.
# - We will then discuss a trick for making the print-out nicer.
# - Finally, we will discuss how the best approach would be to write a class where various parameters (e.g. number of layers, dropout probabilities, etc.) are passed as arguments.

# In[ ]:


from torch.nn import Module
from torch.nn import BatchNorm1d, Linear, ReLU, Dropout


class FCNet(Module):
    """Fully-connected neural network."""


# ### Task 7: Selecting a loss function
# 
# - Binary cross-entropy is about the most common loss function for classification.
#   - Details on this loss function are available in the [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html).
# - Let's instantiate it together.

# In[ ]:


from torch.nn import BCELoss


# ### Task 8: Selecting an optimiser
# 
# While we talked about stochastic gradient descent in the slides, most people use the so-called [Adam optimiser](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).
# 
# You can think of it as a more complex and improved implementation of SGD.

# In[ ]:


# Create an optimiser and give it the model's parameters.
from torch.optim import Adam


# ### Task 9: Writing basic training and validation loops
# 
# - Before we jump in and write these loops, we must first choose an activation function to apply to the model's outputs.
#   - Here we are going to use the softmax activation function: see [the PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
#   - For those of you who've studied physics, you may be remininded of the partition function in thermodynamics.
#   - This activation function is good for classifcation when the result is one of ``A or B or C``.
#     - It's bad if you even want to assign two classification to one images—say a photo of a dog _and_ a cat.
#   - It turns the raw outputs, or logits, into "psuedo probabilities", and we take our prediction to be the most probable class.
# 
# - We will write the training loop together, then you can go ahead and write the (simpler) validation loop.

# In[ ]:


from typing import Dict


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    optimiser: Adam,
    loss_func: BCELoss,
) -> Dict[str, float]:
    """Train ``model`` for once epoch.

    Parameters
    ----------
    model : Module
        The neural network.
    train_loader : DataLoader
        Training dataloader.
    optimiser : Adam
        The optimiser.
    loss_func : BCELoss
        Binary cross-entropy loss function.

    Returns
    -------
    Dict[str, float]
        A dictionary of metrics.

    """


def validate_one_epoch(
    model: Module,
    valid_loader: DataLoader,
    loss_func: BCELoss,
) -> Dict[str, float]:
    """Validate ``model`` for a single epoch.

    Parameters
    ----------
    model : Module
        The neural network.
    valid_loader : DataLoader
        Validation dataloader.
    loss_func : BCELoss
        Binary cross-entropy loss function.

    Returns
    -------
    Dict[str, float]
        Metrics of interest.

    """


# ### Task 10: Training, extracting and plotting metrics
# 
# - Now we can train our model for a specified number of epochs.
#   - During each epoch the model "sees" each training item once.
# - Append the training and validation metrics to a list.
# - Turn them into a ``pandas.DataFrame``
#   - Note: You can turn a ``List[Dict[str, float]]``, say ``my_list`` into a ``DataFrame`` with ``DataFrame(my_list)``.
# - Use Matplotlib to plot the training and validation metrics as a function of the number of epochs.
# 
# We will begin the code block together before you complete it independently.  
# After some time we will go through the solution together.

# In[ ]:


epochs = 3

for _ in range(epochs):
    pass


# ### Task 11: Visualise some results
# 
# Let's do this part together—though feel free to make a start on your own if you have completed the previous exercises.

# In[ ]:


import matplotlib.pyplot as plt


# ### Bonus: Run the net on 'new' inputs
# 
# We have built and trained a net, and evaluated and visualised its performance. However, how do we now utilise it going forward?
# 
# Here we construct some 'new' input data and use our trained net to infer the species. Whilst this is relatively straightforward there is still some work required to transform the outputs from the net to a meaningful result.

# In[ ]:


from torch import no_grad

# Construct a tensor of inputs to run the model over

# Place model in eval mode and run over inputs with no_grad

# Print the raw output from the net

# Transform the raw output back to human-readable format

