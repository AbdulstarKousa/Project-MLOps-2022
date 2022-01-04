# ============================== Import
# import torch
# from torch import nn   # nn.Module
# import torch.nn.functional as F #  F.sigmoid(self.hidden(x)) 
# import numpy as np
# torch.manual_seed(7)
# from torch import optim
# import matplotlib.pyplot as plt


# ============================== Torch functions
""" shape """
# tensor = torch.randn((1, 5))
# tensor.shape

"""
random normal: with mean 0 and std 1
"""
# features = torch.randn((1, 5))
# weights = torch.randn_like(features)
# bias =  torch.randn((1,1))

""" sum """
# torch.sum(features * weights) + bias
# (features * weights).sum() + bias

""" Change the shape """
# weights.shape
# weights.view(features.shape[1], features.shape[0])
# weights.resize_(features.shape[1], features.shape[0])
# weights.reshape(features.shape[1], features.shape[0]) 

"""
matrix multiplications:
    In general, 
    you'll want to use matrix multiplications 
    since they are more efficient and accelerated 
    using modern libraries and high-performance computing on GPUs.
"""
# torch.mm(features, weights.view(features.shape[1], features.shape[0])) + bias

""" Numpy in out """
# torch.from_numpy(np.array([1,2]))
# torch.tensor(np.array([1,2]))
# features.numpy()

""" change type """
# equals.type(torch.FloatTensor)

""" item """
# x.item()


# ============================== Working With Images
""" helper """
# import helper
# helper.imshow(image[0,:]);

""" squeeze """ 
# images[1].numpy().squeeze()

""" imshow: Gray scale """
# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r') 

""" image reshape """
# images.view(images.shape[0], -1)

""" iter """
# dataiter = iter(trainloader)
# images, labels = dataiter.next()




# ============================== Activations
""" Activation """
# nn.Sigmoid()
# nn.Softmax(dim=1)
# F.sigmoid(self.hidden(x))
# F.softmax(self.output(x), dim=1)
# F.log_softmax(self.fc4(x), dim=1)


# ============================== Layers 
""" Layers """
# nn.Linear(784, 256)

""" Access Model Layer weight and bias """
# model.fc1.weight
# model.fc1.bias
# model.fc1.weight.data
# model.fc1.bias.data
# model[0].weight # Sequential with no keys

""" Model Layer weight and bias """
# Model.fc1.bias.data.fill_(0)
# model.fc1.weight.data.normal_(std=0.01)


# ============================== Module 
""" Module """
# class className (nn.Module):
#     def __init__(self):
#         super().__init__()
#         # write the structure, F.ex:
#         self.hidden = nn.Linear(784, 256)
#         self.sigmoid = nn.Sigmoid()
#         self.output = nn.Linear(256, 10)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # forward pass using the input x, F.ex:
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
#         return x

""" Sequential """
## without keys
# model = nn.Sequential(
#         nn.Linear(input_size, hidden_sizes[0]),
#         nn.ReLU(),
#         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#         nn.ReLU(),
#         nn.Linear(hidden_sizes[1], output_size),
#         nn.Softmax(dim=1)
#         )

## with keys
# from collections import OrderedDict
# model = nn.Sequential(
#             OrderedDict([
#                 ('fc1', nn.Linear(input_size, hidden_sizes[0])),
#                 ('relu1', nn.ReLU()),
#                 ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
#                 ('relu2', nn.ReLU()),
#                 ('output', nn.Linear(hidden_sizes[1], output_size)),
#                 ('softmax', nn.Softmax(dim=1))
#             ])
#         )



# ============================== Training
"""
In short: 
    - Forward pass to calculate the *score (logits)* 
    - Use the logits to calculate the *loss*
    - Use backpropagation on loss to calculate *gradients*
    - Use the optimizer to *update weights*

The general process with PyTorch:
    - Make a forward pass through the network 
    - Use the network output to calculate the loss
    - Perform a backward pass through the network 
      with loss.backward() to calculate the gradients
    - Take a step with the optimizer to update the weights
"""

""" Autograd:
    Automatically calculating the gradients of tensors.
"""
# requires_grad = True
# x.requires_grad_(True)
# torch.no_grad()
# with torch.no_grad(): y = x * 2
# torch.set_grad_enabled(True|False)
# y.grad_fn
# x.grad
# z.backward() 

""" Gradient Descent:
    Algorithm to find the minimum loss.
"""

""" Backpropagation: 
    An application of the chain rule from calculus.
"""

""" Learning rate:
    Steps to update the weights
"""

""" Logits (Score):
    the raw output of the model
"""

""" Loss functions (criterion) 
    measure how bad the network is 
"""
# nn.CrossEntropyLoss
# nn.LogSoftmax()
# nn.NLLLoss()

""" Optimizer:
    to update the weights
    requires lr and parameters
"""
# optimizer = optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent
# optimizer = optim.Adam(model.parameters(), lr=0.003) 
# optimizer.zero_grad() # clear
# optimizer.step() # update

""" epoch:
    go through the whole data set
"""

""" batch:
    one subset of the data sent within an epoch
"""

""" basic training loop (no validation or dropout):
    below is the general workflow
"""
# - define model
# - define criterion
# - define optimizer
# - for epoch in epochs:
#     for batch in data loader:
#         - clear (.zero_grad())
#         - logits (forward pass)
#         - loss (criterion(output, labels))
#         - gradients (.backward())
#         - update (.step())



# ============================== Validation
""" overfitting 
    Too well on the training data

    To test for overfitting while training, we
    measure the performance on data not in the training set 
    called the validation set. 
"""

""" early-stopping:    
    to use the version of the
    model with the lowest validation loss.
    
    can be used to avoid overfitting
    
    In practice, you'd save the model 
    frequently as you're training then later
    choose the model with the lowest validation loss.
"""

""" dropout:
    The most common method to reduce overfitting 
    (outside of early-stopping) is dropout, where we
    randomly drop input units.

    During training we want to use dropout to prevent overfitting, 
    but during inference we want to use the entire network. 
    So, we need to turn off dropout during validation, testing, 
    and whenever we're using the network to make predictions. 
    To do this, you use model.eval()
    You can turn dropout back on by setting the
    model to train mode with model.train()
"""
# self.dropout = nn.Dropout(p=0.2) # init
# x = self.dropout(F.relu(self.fc1(x))) # forward


""" Report
    Typically this is just accuracy, the
    percentage of classes the network predicted correctly. Other options are precision and recall and
    top-5 error rate
"""
# ps.topk # This returns the highest values
# ps.topk(1) # This returns a tuple of the top-k values and the top-k indices.
# equals = top_class == labels.view(*top_class.shape)


""" Validation:
    In general, the pattern for the validation loop will look like this, 
    where you turn off gradients, 
    set the model to evaluation mode, 
    calculate the validation loss and metric, 
    then set the model back to train mode
"""
# # turn off gradients
# with torch.no_grad():
#     # set model to evaluation mode
#     model.eval()
# 
#     # validation pass here
#     for images, labels in testloader:
#     ...
# 
# # set model back to train mode
# model.train()

""" Inference: 
    Predection
    Use model.eval()
"""

""" Advance training loop with Validation:
    see Udacity 
"""
# found at 
# Udacity/intro-to-pytorch/Part 5 - Inference and Validation (Solution).ipynb


""" Advance training loop with Validation and dropout:
    see Udacity 
"""
# found at 
# Udacity/intro-to-pytorch/Part 5 - Inference and Validation (Solution).ipynb




# ============================== Save and Load
""" exact architecture
    Loading the state dict works only if the model architecture 
    is exactly the same as the checkpoint architecture
"""
# model = yourModel
# torch.save(model.state_dict(), 'checkpoint.pth')
# state_dict = torch.load('checkpoint.pth')
# model.load_state_dict(state_dict)

""" more dynamic:
    build a dynamic model (input, out, [hidden])
    save (input, out, [hidden]) together with model.state_dict() in dict   
"""
# checkpoint = {
#     'input_size': 784,
#     'output_size': 10,
#     'hidden_layers': [each.out_features for each in model.hidden_layers],
#     'state_dict': model.state_dict()
#     }
# torch.save(checkpoint, 'checkpoint.pth')

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = YourModel(  # this is assuming that YourModel takes these args 
#             checkpoint['input_size'],
#             checkpoint['output_size'],
#             checkpoint['hidden_layers']
#             )
#     model.load_state_dict(checkpoint['state_dict'])
#     return model

# model = load_checkpoint('checkpoint.pth')
# print(model)