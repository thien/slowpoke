# -*- coding: utf-8 -*-
import torch
import random
from torch.autograd import Variable

class dynamicNet(torch.nn.Module):
    def __init__(self, inputDimension, hiddenDimension, outputDimension):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(dynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(inputDimension, hiddenDimension)
        self.middle_linear = torch.nn.Linear(hiddenDimension, hiddenDimension)
        self.output_linear = torch.nn.Linear(hiddenDimension, outputDimension)

    def forward_tempss(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        hidden_relu = self.input_linear(x).clamp(min=0)

        for _ in range(random.randint(0, 3)):
        	# 
            hidden_relu = self.middle_linear(hidden_relu).clamp(min=0)

        # let's see what it does.
        y_pred = self.output_linear(hidden_relu)
        return y_pred

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.
        """
        hidden_relu = self.input_linear(x).clamp(min=0)
        print("--------------------------------------------------------------------")
        print("start",hidden_relu)
        for sk in range(0, 3):
            hidden_relu = self.middle_linear(hidden_relu).clamp(min=0)
            print(sk,hidden_relu)
        y_pred = self.output_linear(hidden_relu)
        return y_pred




# -------------------------------------------------------------------------------

# batchSize is batch size; inputDimension is input dimension;
# hiddenDimension is hidden dimension; outputDimension is output dimension.
batchSize, inputDimension, hiddenDimension, outputDimension = 1, 32, 10, 3

# This represents the number of trials
trials = 43

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(batchSize, inputDimension))
y = Variable(torch.randn(batchSize, outputDimension), requires_grad=False)

# Construct our model by instantiating the class defined above
model = dynamicNet(inputDimension, hiddenDimension, outputDimension)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
# criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# loss = False
k = 0

y_pred = model(x)
print(y_pred)
# for t in range(trials):
#     # Forward pass: Compute predicted y by passing x to the model
#     y_pred = model(x)
#     # print(y_pred)
#     # Compute and print loss
#     # loss = criterion(y_pred, y)
#     k = t
#     # print(t, loss.data[0])

#     # Zero gradients, perform a backward pass, and update the weights.
#     # optimizer.zero_grad()
#     # loss.backward()
#     # print(y_pred)
#     # optimizer.step()
# # print(k, loss.data[0])