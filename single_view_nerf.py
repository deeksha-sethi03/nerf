# -*- coding: utf-8 -*-
"""cis580fall2023_projB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x9ueyXWLFBT5OphW5TNKveSGE5e0urLM

## CIS 580, Machine Perception, Fall 2023
### Homework 5
#### Due: December 22 2023, 11:59pm ET

Instructions: Create a folder in your Google Drive and place inside this .ipynb file. Open the jupyter notebook with Google Colab. Refrain from using a GPU during implementing and testing the whole thing. You should switch to a GPU runtime only when performing the final training (of the 2D image or the NeRF) to avoid GPU usage runouts.

### Part 1: Fitting a 2D Image
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import gdown
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""We first download the image from the web. We normalize the image so the pixels are in between the range of [0,1]."""

url = "https://drive.google.com/file/d/1rD1aaxN8aSynZ8OPA7EI3G936IF0vcUt/view?usp=sharing"
gdown.download(url=url, output='starry_night.jpg', quiet=False, fuzzy=True)

# Load painting image
painting = imageio.imread("starry_night.jpg")
painting = torch.from_numpy(np.array(painting, dtype=np.float32)/255.).to(device)
height_painting, width_painting = painting.shape[:2]

"""1.1 Complete the function positional_encoding()"""

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """

    results = []
    if incl_input:
        results.append(x)


    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.

    # N, L, 2, D


    for i in range(num_frequencies):
      tempsin = torch.sin(2**i * torch.tensor(math.pi) * x)
      tempcos = torch.cos(2**i * torch.tensor(math.pi) * x)
      results.append(tempsin)
      results.append(tempcos)







    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)

"""1.2 Complete the class model_2d() that will be used to fit the 2D image.

"""

class model_2d(nn.Module):

    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        # for autograder compliance, please follow the given naming for your layers
        self.input_dim = num_frequencies * 4 + 2
        self.out_dim = 3 #RGB
        self.layer_in = nn.Linear(self.input_dim, filter_size)
        self.layer = nn.Linear(filter_size, filter_size)
        self.layer_out = nn.Linear(filter_size, self.out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        #############################  TODO 1(b) END  ##############################

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################
        y = self.relu(self.layer_in(x))
        y = self.relu(self.layer(y))
        y = self.sigmoid(self.layer_out(y))


        #############################  TODO 1(b) END  ##############################
        return y

def normalize_coord(height, width, num_frequencies=6):

    """
    Creates the 2D normalized coordinates, and applies positional encoding to them

    Args:
    height (int): Height of the image
    width (int): Width of the image
    num_frequencies (optional, int): The number of frequencies used in
      the positional encoding (default: 6).

    Returns:
    (torch.Tensor): Returns the 2D normalized coordinates after applying positional encoding to them.
    """

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them


    coordinates = torch.cartesian_prod(torch.Tensor(np.arange(height)), torch.Tensor(np.arange(width)))

    coordinates[:, 0] /= height
    coordinates[:, -1] /= width
    # embedded_coordinates = coordinates

    embedded_coordinates = positional_encoding(coordinates, num_frequencies)



    #############################  TODO 1(c) END  ############################

    return embedded_coordinates

"""You need to complete 1.1 and 1.2 first before completing the train_2d_model function. Don't forget to transfer the completed functions from 1.1 and 1.2 to the part1.py file and upload it to the autograder.

Fill the gaps in the train_2d_model() function to train the model to fit the 2D image.
"""

def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer

    optimizer = torch.optim.Adam(model2d.parameters(), lr = lr)

    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them

    embedded_coordinates = normalize_coord(height, width, num_frequencies).to(device)
    # print(normalized_coordinates.size())
    # embedded_coordinates = positional_encoding(normalized_coordinates, num_frequencies).to(device)

    coordinates = torch.cartesian_prod(torch.Tensor(np.arange(height)), torch.Tensor(np.arange(width)))

    coordinates[:, 0] /= height
    coordinates[:, -1] /= width
    normalized_coordinates = coordinates



    criterion = torch.nn.MSELoss()

    #############################  TODO 1(c) END  ############################

    for i in range(iterations+1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration

        pred = model2d(embedded_coordinates).reshape((height, width, 3))

        loss = criterion(pred, test_img)

        loss.backward()

        optimizer.step()

        # Compute mean-squared error between the predicted and target images. Backprop!



        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr

            Rsquare = torch.max(normalized_coordinates) ** 2

            psnr = 10 * torch.log10(Rsquare/loss)


            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print('Done!')
    torch.save(model2d.state_dict(),'model_2d_' + str(num_frequencies) + 'freq.pt')
    plt.imsave('van_gogh_' + str(num_frequencies) + 'freq.png',pred.detach().cpu().numpy())
    return pred.detach().cpu()

"""Train the model to fit the given image without applying positional encoding to the input, and by applying positional encoding of two different frequencies to the input; L = 2 and L = 6."""

_ = train_2d_model(test_img=painting, num_frequencies=2, device=device)