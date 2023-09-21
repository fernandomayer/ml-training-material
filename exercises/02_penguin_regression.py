#!/usr/bin/env python
# coding: utf-8

# # Exercise 2: Penguin regression with PyTorch
# 
# <img src="https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png" width="750" />
# 
# 
# Artwork by @allison_horst
# 
# In this exercise, we will again use the [``palmerpenguins``](https://github.com/mcnakhaee/palmerpenguins) data to continue our exploration of PyTorch.
# 
# We will use the same dataset object as before, but this time we'll take a look at a regression problem: predicting the mass of a penguin given other physical features.

# ### Task 1: look at the data
# In the following code block, we import the ``load_penguins`` function from the ``palmerpenguins`` package.
# 
# - Load the penguin data as you did before.
# - This time, consider which features we might like to use to predict a penguin's mass.

# In[ ]:


from palmerpenguins import load_penguins

# Insert code here ...


# The features we will use to estimate the mass are:
# - ...
# - ...
# - ...

# ### Task 2: creating a ``torch.utils.data.Dataset``
# 
# As before, we need to create PyTorch ``Dataset`` objects to supply data to our neural network.  
# Since we have already created and explored the ``PenguinDataset`` class there is nothing else to do here.

# ### Task 3: Obtaining training and validation datasets.
# 
# - Instantiate the penguin dataloader.
#   - Make sure you supply the correct column titles for the features and the targets.
#   - Remember, the target is now mass, not the species!
# - Iterate over the dataset
#     - Hint:
#         ```python
#         for features, targets in dataset:
#             # print the features and targets here
#         ```

# In[ ]:


from ml_workshop import PenguinDataset


# ### Task 4: Applying transforms to the data
# 
# As in the previous exercise, the raw inputs and targets need transforming to ``torch.Tensor``s before they can be passed to a neural network.  
# We will again use ``torchvision.transforms.Compose`` to take a list of callable objects and apply them to the incoming data.
# 
# Because the raw units of mass are in grams, the numbers are quite large. This can encumber the model's predictive power. A sensible way to address this is to normalise targets using statistics from the training set. The most common form of normalisation is to subtract the mean and divide by the standard deviation (of the training set). However here, for the sake of simplicity, we will just scale the mass by dividing by the mean of the training set.
# 
# Note that this means that the model will now be trained to predict masses as fractions of the training mean.
# 
# We grab the mean of the training split in the following cell.

# In[ ]:


train_set = PenguinDataset(features, ["body_mass_g"], train=True)

training_mean = train_set.split.body_mass_g.mean()


# Now we create our real training and validation set, and supply transforms as before.

# In[ ]:


from torchvision.transforms import Compose

# Let's apply the transfroms we need to the PenguinDataset to get out inputs
# targets as Tensors.


# ### Task 5: Creating ``DataLoaders``—Again!
# 
# As before, we wrap our ``Dataset``s in ``DataLoader`` before we proceed.

# In[ ]:


from torch.utils.data import DataLoader

# Create training and validation DataLoaders.


# ### Task 6: Creating a neural network in PyTorch
# 
# Previously we created our neural network from scratch, but doing this every time we need to solve a new problem is cumbersome.  
# Many projects working with the ICCS have codes where the numbers of layers, layer sizes, and other parts of the models are hard-coded from scratch every time!
# 
# The result is ungainly, non-general, and heavily-duplicated code. Here, we are going to shamelessly punt Jim Denholm's Python repo, [``TorchTools``](https://github.com/jdenholm/TorchTools), which contains generalisations of many commonly-used PyTorch tools, to save save us some time.
# 
# Here, we can use the ``FCNet`` model, whose documentation lives [here](https://jdenholm.github.io/TorchTools/models.html). This model is a fully-connected neural network with various options for dropout, batch normalisation, and easily-modifiable layers.
# 
# #### A brief sidebar
# Note: the repo is pip-installable with
# ```bash
# pip install git+https://github.com/jdenholm/TorchTools.git
# ```
# but has already been installed for you in the requirements of this workshop package.
# 
# It is useful to know you can install Python packages from GitHub using pip. To install specific versions you can use:
# ```bash
# pip install git+https://github.com/jdenholm/TorchTools.git@v0.1.0
# ```
# (The famous [segment anything model](https://github.com/facebookresearch/segment-anything) (SAM) published by Facebook Research was released in this way.)
# 
# One might argue that this is a much better way of making one-off codes available, for example academic codes which might accompany papers, rather than using the global communal package index PyPI.

# ##### Back to work: let's instantiate the model.

# In[ ]:


from torch_tools import FCNet

# model =


# ### Task 7: Selecting a loss function
# 
# The previous loss function we chose was appropriate for classification, but _not_ regression.  
# Here we'll use the mean-squared-error loss, which is more appropriate for regression.

# In[ ]:


from torch.nn import MSELoss

# loss_func = ...


# ### Task 8: Selecting an optimiser
# 
# ``Adam`` is regarded as the king of optimisers: let's use it again, but this time with a learning rate.
# 
# [Adam docs](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

# In[ ]:


from torch.optim import Adam

# Create an optimiser and give it the model's parameters.


# ### Task 9: Writing basic training and validation loops
# 
# 
# As before, we will write the training loop together and you can then continue with the validation loop.
# 

# In[ ]:


from typing import Dict
from torch.nn import Module


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    optimiser: Adam,
    loss_func: MSELoss,
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
    loss_func : MSELoss
        Mean-squared-error loss.
    Returns
    -------
    Dict[str, float]
        A dictionary of metrics.

    """


def validate_one_epoch(
    model: Module,
    valid_loader: DataLoader,
    loss_func: MSELoss,
) -> Dict[str, float]:
    """Validate ``model`` for a single epoch.

    Parameters
    ----------
    model : Module
        The neural network.
    valid_loader : DataLoader
        Validation dataloader.
    loss_func : MSELoss
        Mean-squared-error loss.

    Returns
    -------
    Dict[str, float]
        Metrics of interest.

    """


# ### Task 10: Training and extracting metrics
# 
# - Now we can train our model for a specified number of epochs.
#   - During each epoch the model "sees" each training item once.
# - Append the training and validation metrics to a list.
# - Turm them into a ``pandas.DataFrame``
#   - Note: You can turn a ``List[Dict[str, float]]``, say ``my_list`` into a ``DataFrame`` with ``DataFrame(my_list)``.

# In[ ]:


epochs = 3

for _ in range(epochs):
    pass


# ### Task 11: Plotting metrics
# 
# - Use Matplotlib to plot the training and validation metrics as a function of the number of epochs.
# - Does this allow us to interpret performance?

# In[ ]:


import matplotlib.pyplot as plt

