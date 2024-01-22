"""### Part 2: Fitting a 3D Image"""

import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import imageio.v2 as imageio
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

url = "https://drive.google.com/file/d/13eBK_LWxs4SkruFKH7glKK9jwQU1BkXK/view?usp=sharing"
gdown.download(url=url, output='lego_data.npz', quiet=False, fuzzy=True)

"""Here, we load the data that is comprised by the images, the R and T matrices of each camera position with respect to the world coordinates and the intrinsics parameters K of the camera."""

# Load input images, poses, and intrinsics
data = np.load("lego_data.npz")

# Images
images = data["images"]

# Height and width of each image
height, width = images.shape[1:3]

# Camera extrinsics (poses)
poses = data["poses"]
poses = torch.from_numpy(poses).to(device)
print(poses.shape)

# Camera intrinsics
intrinsics = data["intrinsics"]
intrinsics = torch.from_numpy(intrinsics).to(device)

# Hold one image out (for test).
test_image, test_pose = images[101], poses[101]
test_image = torch.from_numpy(test_image).to(device)

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

plt.imshow(test_image.detach().cpu().numpy())
plt.show()

print(data)

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


"""2.1 Complete the following function that calculates the rays that pass through all the pixels of an HxW image"""

def get_rays(height, width, intrinsics, w_R_c, w_T_c):

    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    #############################  TODO 2.1 BEGIN  ##########################

    ray_origins = torch.broadcast_to(w_T_c, (height, width, 3))

    # coordinates = torch.cartesian_prod(torch.Tensor(np.arange(height)), torch.Tensor(np.arange(width)))

    # coordinates = torch.cat((coordinates, torch.ones(coordinates.size()[0], 1)), dim=1).to(device)

    H, W = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy')

    coordinates = torch.vstack((H.flatten(), W.flatten(), torch.ones_like(H.flatten()))).float().to(device)

    ray_directions_interim = torch.linalg.inv(intrinsics) @ coordinates

    ray_directions = ((w_R_c @ ray_directions_interim).T).reshape((height, width, 3))

    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

"""Complete the next function to visualize how is the dataset created. You will be able to see from which point of view each image has been captured for the 3D object. What we want to achieve here, is to being able to interpolate between these given views and synthesize new realistic views of the 3D object."""

def plot_all_poses(poses):

    #############################  TODO 2.1 BEGIN  ############################

    origins, directions = [], []

    for current_pose in poses:
      current_pose = torch.Tensor(current_pose).to(device)
      ray_origins, ray_directions = get_rays(height, width, intrinsics, current_pose[:-1, :-1], current_pose[:-1, -1])
      origins.append(ray_origins.cpu())
      directions.append(ray_directions.cpu())

    origins = torch.cat(origins)
    directions = torch.cat(directions)



    #############################  TODO 2.1 END  ############################

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(),
                  origins[..., 1].flatten(),
                  origins[..., 2].flatten(),
                  directions[..., 0].flatten(),
                  directions[..., 1].flatten(),
                  directions[..., 2].flatten(), length=0.12, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()

plot_all_poses(data['poses'])

"""2.2 Complete the following function to implement the sampling of points along a given ray."""

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################

    i_values = torch.linspace(near, far, samples)

    ray_points = torch.zeros((ray_origins.size()[0], ray_origins.size()[1], samples, 3))
    depth_points = torch.zeros((ray_origins.size()[0], ray_origins.size()[1], samples))



    # ray_directions = ray_directions.view((ray_directions.size()[0]*ray_directions.size()[1], 3))
    # ray_origins = ray_origins.view((ray_origins.size()[0]*ray_origins.size()[1], 3))

    for i in range(samples):
      ti = near + (i) / samples * (far - near)
      ray_endpoint = ray_origins + ti * ray_directions
      ray_points[:, :, i, :] = ray_endpoint
      depth_points[:, :, i] = ti

    ray_points, depth_points = ray_points.to(device), depth_points.to(device)

    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points

"""2.3 Define the network architecture of NeRF along with a function that divided data into chunks to avoid memory leaks during training. """


class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        # for autograder compliance, please follow the given naming for your layers
        self.input_dim = 6 * num_x_frequencies + 3
        self.inter_dim = filter_size + 6 * num_x_frequencies + 3
        self.out_dim = filter_size + 6 * num_d_frequencies + 3

        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(self.input_dim, filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),
            'layer_6': nn.Linear(self.inter_dim, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_s': nn.Linear(filter_size, 1),
            'layer_9': nn.Linear(filter_size, filter_size),
            'layer_10': nn.Linear(self.out_dim, filter_size//2),
            'layer_11': nn.Linear(filter_size//2, 3)
        })
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        # example of forward through a layer: y = self.layers['layer_1'](x)


        y = self.relu(self.layers['layer_1'](x))
        y = self.relu(self.layers['layer_2'](y))
        y = self.relu(self.layers['layer_3'](y))
        y = self.relu(self.layers['layer_4'](y))
        y = self.relu(self.layers['layer_5'](y))
        yx = torch.cat((y, x), dim = -1)
        yx = self.relu(self.layers['layer_6'](yx))
        yx = self.relu(self.layers['layer_7'](yx))
        yx = self.layers['layer_8'](yx)
        sigma = self.layers['layer_s'](yx)
        yx = self.layers['layer_9'](yx)
        yd = torch.cat((yx, d), dim = -1)
        yd = self.relu(self.layers['layer_10'](yd))
        rgb = self.sigmoid(self.layers['layer_11'](yd))

        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################

    ray_directions = ray_directions / torch.norm(ray_directions, dim=2).unsqueeze(dim=2)
    ray_directions = torch.broadcast_to(ray_directions.unsqueeze(dim=2), ray_points.size()).reshape(-1, 3)
    ray_directions_embedded = positional_encoding(ray_directions, num_d_frequencies)
    ray_points_embedded = positional_encoding(ray_points.reshape(-1, 3), num_x_frequencies)
    ray_points_batches = get_chunks(ray_points_embedded)
    ray_directions_batches = get_chunks(ray_directions_embedded)

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

"""2.4 Compute the compositing weights of samples on camera ray and then complete the volumetric rendering procedure to reconstruct a whole RGB image from the sampled points and the outputs of the neural network."""

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    #############################  TODO 2.4 BEGIN  ############################

    delta = depth_points - torch.roll(depth_points, shifts=1, dims=-1)
    delta = delta[..., 1:].to(device)
    deltaN = torch.full((delta.shape[0], delta.shape[1], 1), 1e9, device=device)
    delta = torch.concat((delta,deltaN),-1).to(device)

    relu = nn.ReLU()
    product_term = torch.exp(-1 * relu(s.to(device)) * delta).to(device)
    T = torch.cumprod(product_term, dim=-1).to(device)
    T = torch.cat([torch.ones_like(T[..., :1], device=device), T[..., :-1]], dim=-1)

    C = torch.mul(T.unsqueeze(-1), (1 - product_term.unsqueeze(-1)) * rgb.to(device))
    rec_image = C.sum(dim=2).to(device)

    #############################  TODO 2.4 END  ############################

    return rec_image

"""To test and visualize your implementation, independently of the previous and next steps of the
NeRF pipeline, you can load the sanity_volumentric.pt file, run your implementation of the volumetric function and expect to see the figure provided in the handout.

"""

url = "https://drive.google.com/file/d/1ag6MqSh3h4KY10Mcx5fKxt9roGNLLILK/view?usp=sharing"
gdown.download(url=url, output='sanity_volumentric.pt', quiet=False, fuzzy=True)
rbd = torch.load('sanity_volumentric.pt')

r = rbd['rgb']
s = rbd['sigma']
depth_points = rbd['depth_points']
rec_image = volumetric_rendering(r, s, depth_points)

plt.figure(figsize=(10, 5))
plt.imshow(rec_image.detach().cpu().numpy())
plt.title(f'Volumetric rendering of a sphere with $\\sigma={0.2}$, on blue background')
plt.show()

"""2.5 Combine everything together. Given the pose position of a camera, compute the camera rays and sample the 3D points along these rays. Divide those points into batches and feed them to the neural network. Concatenate them and use them for the volumetric rendering to reconstructed the final image."""

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):

    #############################  TODO 2.5 BEGIN  ############################

    #compute all the rays from the image
    ray_origins, ray_directions = get_rays(height, width, intrinsics, pose[:-1, :-1], pose[:-1, -1])

    #sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_points_batches,  ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    rgb_batches, sigma_batches = zip(*[model(ray_point, ray_direction) for ray_point, ray_direction in zip(ray_points_batches, ray_directions_batches)])

    rgb = torch.cat(rgb_batches).view(height, width, samples, 3)

    sigma = torch.cat(sigma_batches).view(height, width, samples)

    rec_image = volumetric_rendering(rgb, sigma, depth_points)

    #############################  TODO 2.5 END  ############################

    return rec_image

