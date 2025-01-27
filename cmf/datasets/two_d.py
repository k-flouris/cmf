import numpy as np

import torch

from scipy.stats import vonmises

import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

from .supervised_dataset import SupervisedDataset

from scipy.stats import beta

def hemisphere_dataset(size, d_prime=2, d=3, noise_level=0.01):
    """
    Generate a tighter Hemisphere dataset with uniform noise for a compact structure.
    
    Parameters:
    - size: Number of samples
    - d_prime: Intrinsic dimension (hemisphere dimension)
    - d: Ambient dimension (should be greater than or equal to d_prime + 1)
    - noise_level: Scale of the uniform noise to keep the pattern tight
    
    Returns:
    - data: Generated samples in ambient space (shape: [size, d])
    """
    # Step 1: Sample spherical coordinates for a hemisphere shape
    theta1 = beta.rvs(5, 5, size=size) * (np.pi / 2)  # Keeping theta1 values toward the equator
    other_thetas = np.random.uniform(0, np.pi, size=(size, d_prime - 1))
    
    # Step 2: Convert to Cartesian coordinates in Rd_prime+1 (upper hemisphere)
    x = np.ones((size, d_prime + 1))
    x[:, 0] = np.cos(theta1)
    for i in range(1, d_prime + 1):
        angle_product = np.prod(np.sin(other_thetas[:, :i - 1]), axis=1) if i > 1 else 1
        x[:, i] = angle_product * (np.cos(other_thetas[:, i - 1]) if i < d_prime else np.sin(other_thetas[:, i - 2]))
    
    # Step 3: Apply random isometric embedding in ambient space with uniform noise
    random_matrix = np.random.randn(d, d_prime + 1)
    q, _ = np.linalg.qr(random_matrix)
    data = np.dot(x, q.T)
    
    # Add uniform noise to create a tighter effect
    noise = np.random.uniform(-noise_level, noise_level, size=(size, d))
    data += noise
    return data

def sinusoid_dataset(size, d_prime=1, d=3, sigma_m=0.5, noise_level=0.001):
    """
    Generate a tighter sinusoidal dataset with uniform noise for a compact snake-like pattern.
    
    Parameters:
    - size: Number of samples
    - d_prime: Intrinsic dimension (latent space dimension)
    - d: Ambient dimension
    - sigma_m: Variance for latent Gaussian variables
    - noise_level: Scale of the uniform noise for ambient coordinates
    
    Returns:
    - data: Generated samples in ambient space (shape: [size, d])
    """
    # Step 1: Sample latent variables z from Gaussian distribution
    z = np.random.normal(0, np.sqrt(sigma_m), size=(size, d_prime))
    
    # Step 2: Generate ambient coordinates with higher frequency sinusoidal transformations
    a_j = np.random.uniform(3, 4, size=(d - d_prime, d_prime))  # Increase frequency range for tighter pattern
    ambient_coords = np.sin(np.dot(z, a_j.T)) + np.random.uniform(-noise_level, noise_level, size=(size, d - d_prime))
    
    # Step 3: Concatenate ambient coordinates with latent variables to form samples in Rd
    data = np.hstack([ambient_coords, z])
    return data


# Redefine the river_dataset_tight_uniform function for a tighter, more controlled river pattern with uniform noise
def river_dataset(size, a=4, noise_level=0.02):
    """
    Generates a tighter River dataset with uniform noise for a more compact meandering path.
    
    Parameters:
    - size: Number of samples.
    - a: Controls the sinusoidal frequency (higher for a tighter curve).
    - noise_level: Scale of the uniform noise for tight appearance.
    
    Returns:
    - data: Generated River dataset samples.
    """
    # Generate x2 uniformly across the interval
    x2 = np.linspace(-2, 2, size)
    
    # Apply diffeomorphism to generate x1 based on x2 with higher frequency
    x1 = np.sin(a * x2)

    # Stack x1 and x2 to form the data points
    data = np.vstack((x1, x2)).T
    
    # Apply uniform noise for a tighter appearance
    noise = np.random.uniform(-noise_level, noise_level, size=(size, 2))
    data += noise
    return data


# Modified from https://github.com/jhjacobsen/invertible-resnet/blob/278faffe7bf25cd7488f8cd49bf5c90a1a82fc0c/models/toy_data.py#L8 
def get_2d_data(data, size):
    if data == "hemisphere-2-6":
        data = hemisphere_dataset(size, d_prime=2, d=6)

    elif data == "sinusoid-1-3":
        data = sinusoid_dataset(size, d_prime=1, d=3,  noise_level=0.1, sigma_m=0.1)  
        
    elif data == "sinusoid-1-6":
        data = sinusoid_dataset(size, d_prime=1, d=6,  noise_level=0.1, sigma_m=0.1)  
        
    elif data == "river":
                data = river_dataset(size)
    elif data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=size, noise=1.0)[0]
        data = data[:, [0, 2]]
        data /= 5

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=size, factor=.5, noise=0.08)[0]
        data *= 3

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = size // 4
        n_samples1 = size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        # Add noise
        data = X + np.random.normal(scale=0.08, size=X.shape)

    elif data == "8gaussians":
        dim = 2
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        for i in range(len(centers)):
          for k in range(dim-2):
            centers[i] = centers[i]+(0,)

        data = []
        for i in range(size):
            point = np.random.randn(dim) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data)
        data /= 1.414

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        data = x

    elif data == "checkerboard":
        x1 = np.random.rand(size) * 4 - 2
        x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1)
        data *= 2

    elif data == "fuzzy-line":
        x = np.random.rand(size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1) 
        noise_ = np.random.rand(size)*0.5
        data += np.stack((noise_, -noise_), 1)
        
    elif data == "linein3d":
        x = np.random.rand(size) * 5 - 2.5
        y = x
        z = np.zeros_like(x)
        data = np.stack((x, y, z ), 1) 
        noise_ = np.random.rand(size)*0.5
        data += np.stack((noise_, -noise_, np.zeros_like(noise_)), 1)
    
    elif data == "cross":
        x1= np.random.rand(size) * 5 - 2.5
        
        x2 = np.empty(size)
        x2[:size//2] =  x1[:size//2]
        x2[size//2:] = -x1[size//2:] 
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)    
     
        
    elif data == "pure-line":
        x = np.random.rand(size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1) 
        # noise_ = np.random.rand(size)*0.5
        # data += np.stack((noise_, -noise_), 1)
        
    elif data == "3d-line":
        x = np.random.rand(size) * 5 - 2.5
        y = x
        z = x+y
        data = np.stack((x, y, z ), 1) 
        noise_ = np.random.rand(size)*0.5
        data += np.stack((noise_, noise_, -noise_), 1)
 
    
    elif data == "shifted-line":
        x = np.random.rand(size) * 5 + 2.5
        y = x
        data = np.stack((x, y), 1) 
        noise_ = np.random.rand(size)*0.5
        data += np.stack((noise_, -noise_), 1)
        
    elif data == "box":
        x = np.random.rand(size) * 5 - 2.5
        y = np.random.rand(size) * 5 - 2.5
        data = np.stack((x, y), 1) 

    elif data == "vertical-line":
        x = np.random.rand(size) * 0.1 - 0.05
        y = np.random.rand(size) * 5 - 2.5
        data = np.stack((x, y), 1) 
            
    elif data == "cos":
        x = np.random.rand(size) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)

    elif data == "2uniforms":
        mixture_component = (np.random.rand(size) > 0.5).astype(int)
        x1 = np.random.rand(size) + mixture_component - 2*(1 - mixture_component)
        x2 = 2 * (np.random.rand(size) - 0.5)
        data = np.stack((x1, x2), 1)

    elif data == "2lines":
        x1 = np.empty(size)
        x1[:size//2] = -1.
        x1[size//2:] = 1.
        x1 += 0.01 * (np.random.rand(size) - .5)
        x2 = 2 * (np.random.rand(size) - 0.5)
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "2marginals":
        x1 = np.empty(size)
        x1[:size//2] = -1.
        x1[size//2:] = 1.
        x1 += .5 * (np.random.rand(size) - .5)
        x2 = np.random.normal(size=size)
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "1uniform":
        x1 = np.random.rand(size) - .5
        x2 = np.random.rand(size) - .5
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "annulus":
        rad1 = 2
        rad2 = 1
        theta = 2 * np.pi * np.random.random(size)
        # theta = 2 * np.pi * np.linspace(0,1,size)
        r = np.sqrt(np.random.random(size) * (rad1**2 - rad2**2) + rad2**2)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        data = np.stack((x1, x2), 1)
        
    elif data == "ellipse":
        rad1 = 2
        rad2 = 1
        # theta = 2 * np.pi * np.random.random(size)
        theta = 2 * np.pi * np.linspace(0,1,size)
        # r1 = np.sqrt(np.random.random(size) * (rad1**2)) #NB to make it assymetric from zero
        # r2 = np.sqrt(np.random.random(size) * (rad2**2))
        r1 = np.random.random(size) * rad1
        r2 = np.random.random(size) * rad2
        # x1 = r1 * np.cos(theta) # NB. Original ellipse
        # x2 = r2 * np.sin(theta)
        phi=np.pi/4 # rotation angle
        x1 = r1 * np.cos(theta)*np.cos(phi)-r2*np.sin(theta)*np.sin(phi)
        x2 = r1*np.cos(theta)*np.sin(phi)+r2*np.sin(theta)*np.cos(phi)
        data = np.stack((x1, x2), 1)

    elif data == "2ellipses":
        x1 = np.empty(size)
        x2 = np.empty(size)

                
        # theta = 2 * np.pi * np.random.random(size/2)
        theta = 2 * np.pi * np.linspace(0,1,int(size/2))
        radA1 = 2
        radA2 = 0.2
        factor=10/10
        rA1 = np.power(np.random.random(int(size/2)) * (radA1**factor),1/factor)
        rA2 = np.power(np.random.random(int(size/2)) * (radA2**factor),1/factor)
        radB1 = 2
        radB2 = 0.2       
        rB1 = np.power(np.random.random(int(size/2)) * (radB1**factor),1/factor)
        rB2 = np.power(np.random.random(int(size/2)) * (radB2**factor),1/factor)
        
        phiA=np.pi/2 # rotation angle A
        phiB=np.pi/6 #-np.pi/4 # rotation angle B
        x1[:size//2] = rA1*np.cos(theta)*np.cos(phiA)-rA2*np.sin(theta)*np.sin(phiA)
        x2[:size//2] = rA1*np.cos(theta)*np.sin(phiA)+rA2*np.sin(theta)*np.cos(phiA)
        
        x1[size//2:] = rB1*np.cos(theta)*np.cos(phiB)-rB2*np.sin(theta)*np.sin(phiB)
        x2[size//2:] = rB1*np.cos(theta)*np.sin(phiB)+rB2*np.sin(theta)*np.cos(phiB)
        
        data = np.stack((x1, x2), 1)


    elif data == "sawtooth":
        u = np.random.rand(size)
        branch = u < .5
        x1 = np.zeros(size)
        x1[branch] = -1 - np.sqrt(1 - 2*u[branch])
        x1[~branch] = 1 + np.sqrt(2*u[~branch] - 1)
        x2 = np.random.rand(size)
        data = np.stack((x1, x2), 1)

    elif data == "quadspline":
        u = np.random.rand(size)
        branch = u < .5
        x1 = np.zeros(size)
        x1[branch] = -1 + np.cbrt(2*u[branch] - 1)
        x1[~branch] = 1 + np.cbrt(2*u[~branch] - 1)
        x2 = np.random.rand(size)
        data = np.stack((x1, x2), 1)

    elif data == "split-gaussian":
        x1 = np.random.normal(size=size)
        x2 = np.random.normal(size=size)
        x2[x1 >= 0] += 2
        x2[x1 < 0] -= 2
        data = np.stack((x1, x2), 1)

    elif data == "von-mises-circle":
        theta = vonmises.rvs(1, size=size, loc=np.pi/2)
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        data = np.stack((x1, x2), 1)

    elif data == "von-mises-sphere":
        theta = vonmises.rvs(1, size=size, loc=np.pi/2)
        phi = vonmises.rvs(1, size=size, loc=np.pi/2)/2
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data = np.stack((x1, x2, x3), 1)
        
    elif data == "sphere":
        theta = 2 * np.pi * np.random.random(size)
        phi =  np.pi * np.random.random(size)
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data = np.stack((x1, x2, x3), 1)
        
    elif data == "offcenter-sphere":
        theta = 2 * np.pi * np.random.random(size)
        phi =  np.pi * np.random.random(size)
        x1 = np.cos(theta)*np.sin(phi)+10
        x2 = np.sin(theta)*np.sin(phi)+10
        x3 = np.cos(phi)+10
        data = np.stack((x1, x2, x3), 1)
        
    elif data == "offcenter-spheres":
        theta_A = 2 * np.pi * np.random.random(int(9*size/10))
        phi_A =  np.pi * np.random.random(int(9*size/10))
        
        theta_B = 2 * np.pi * np.random.random(int(size/10))
        phi_B =  np.pi * np.random.random(int(size/10))
        
        x1_A = np.cos(theta_A)*np.sin(phi_A)+10
        x1_B = np.cos(theta_B)*np.sin(phi_B)-2
        x1 = np.concatenate((x1_A, x1_B))

        
        x2_A = np.sin(theta_A)*np.sin(phi_A)+10
        x2_B = np.sin(theta_B)*np.sin(phi_B)-2
        x2 = np.concatenate((x2_A, x2_B))

        
        x3_A = np.cos(phi_A)+10
        x3_B = np.cos(phi_B)-2
        x3 = np.concatenate((x3_A, x3_B))

        
        data = np.stack((x1, x2, x3), 1)
        
    elif data == "randomized-s2inr6":
        theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
        phi = np.pi * np.random.random(size)        # Random polar angle
    
        # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
        x1 = np.cos(theta) * np.sin(phi)
        x2 = np.sin(theta) * np.sin(phi)
        x3 = np.cos(phi)
    
        # Stack these coordinates to form the initial 3D points
        data_s2 = np.stack((x1, x2, x3), axis=1)
    
        # Add three small random components for embedding into 6D space
        x4 = 0.03 * np.random.randn(size)  # Random noise for 4th dimension
        x5 = 0.03 * np.random.randn(size)  # Random noise for 5th dimension
        x6 = 0.03 * np.random.randn(size)  # Random noise for 6th dimension
    
        # Stack the 6D coordinates to form a sphere in R6 with some randomness
        data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
        data = data_r6
        
    elif data == "randomized-s2inr6-001":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.01 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.01 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.01 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6
            
    elif data == "randomized-s2inr6-005":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.05 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.05 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.05 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6

    elif data == "randomized-s2inr6-003-1":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.03 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.03 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 1 * np.ones(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6


    elif data == "randomized-s2inr6-005-0":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.05 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.05 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.00 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6

    elif data == "randomized-s2inr6-001-0":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.01 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.01 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.00 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6
            
    elif data == "randomized-s2inr6-003":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.03 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.03 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.03 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6

    elif data == "randomized-s2inr6-003-0":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.03 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.03 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.00 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6


    elif data == "randomized-s2inr6-000":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.00 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.00 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.00 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6
            
    elif data == "null6d":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = 0*np.cos(theta) * np.sin(phi)
            x2 = 0*np.sin(theta) * np.sin(phi)
            x3 = 0*np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.00 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.00 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.00 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6
    elif data == "randomized-s2inr6-003-0015-0":
            theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
            phi = np.pi * np.random.random(size)        # Random polar angle
        
            # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
            x1 = np.cos(theta) * np.sin(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(phi)
        
            # Stack these coordinates to form the initial 3D points
            data_s2 = np.stack((x1, x2, x3), axis=1)
        
            # Add three small random components for embedding into 6D space
            x4 = 0.03 * np.random.randn(size)  # Random noise for 4th dimension
            x5 = 0.015 * np.random.randn(size)  # Random noise for 5th dimension
            x6 = 0.00 * np.random.randn(size)  # Random noise for 6th dimension
        
            # Stack the 6D coordinates to form a sphere in R6 with some randomness
            data_r6 = np.hstack((data_s2, x4[:, None], x5[:, None], x6[:, None]))
            data = data_r6
            
    elif data == "s4inr6":
        theta = 2 * np.pi * np.random.random(size)  # Uniformly distributed from 0 to 2*pi
        phi = np.pi * np.random.random(size)       # Uniformly distributed from 0 to pi
        psi = 2 * np.pi * np.random.random(size)   # Uniformly distributed from 0 to 2*pi
        
        # Calculate the 4D coordinates for S^3
        x1 = np.sin(psi) * np.sin(phi) * np.cos(theta)
        x2 = np.sin(psi) * np.sin(phi) * np.sin(theta)
        x3 = np.sin(psi) * np.cos(phi)
        x4 = np.cos(psi)
        
        # Stack these coordinates to form the 4D points
        data_s3 = np.stack((x1, x2, x3, x4), axis=1)
        
        u, v, w, t = x1, x2, x3, x4
        denominator = 1 + u**2 + v**2 + w**2 + t**2
        x_1 = u * 2 / denominator
        x_2 = v * 2 / denominator
        x_3 = w * 2 / denominator
        x_4 = t * 2 / denominator
        x_5 = 1 - 2 / denominator  # This is essentially x_5 acting as the embedded coordinate into R5


        # Stack these coordinates to form the 5D points
        data_in_R5 = np.stack((x_1, x_2, x_3, x_4, x_5), axis=1)
        x5=x_5
        u, v, w, t, s = x1, x2, x3, x4, x5
        denominator = 1 + u**2 + v**2 + w**2 + t**2 + s**2
        x_1 = u * 2 / denominator
        x_2 = v * 2 / denominator
        x_3 = w * 2 / denominator
        x_4 = t * 2 / denominator
        x_5 = s * 2 / denominator
        x_6 = 1 - 2 / denominator  # This acts as the embedded coordinate into R6

        # Stack these coordinates to form the 6D points
        data_in_R6 = np.stack((x_1, x_2, x_3, x_4, x_5, x_6), axis=1)
        # Trivially embed this into R5 by adding a zero coordinate
        # data_r5 = np.hstack((data_s3, np.zeros((size, 1))))
        # data_r6 = np.hstack((data_r5, np.zeros((size, 1))))
        data=data_in_R6

    elif data == "s2inr6":
        theta = 2 * np.pi * np.random.random(size)
        phi =  np.pi * np.random.random(size)
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data_s2 = np.stack((x1, x2, x3), 1)
        # Stack these coordinates to form the 4D points
        # data_s3 = np.stack((x1, x2, x3, x4), axis=1)
        u, v, w = x1, x2, x3
        denominator = 1 + u**2 + v**2 + w**2 
        x_1 = u * 2 / denominator
        x_2 = v * 2 / denominator
        x_3 = w * 2 / denominator
        x_4 = 1 - 2 / denominator  # This is essentially x_5 acting as the embedded coordinate into R5
        
        u, v, w, t = x_1, x_2, x_3, x_4
        denominator = 1 + u**2 + v**2 + w**2 + t**2
        x_1 = u * 2 / denominator
        x_2 = v * 2 / denominator
        x_3 = w * 2 / denominator
        x_4 = t * 2 / denominator
        x_5 = 1 - 2 / denominator  # This is essentially x_5 acting as the embedded coordinate into R5


        # Stack these coordinates to form the 5D points
        u, v, w, t, s = x_1, x_2, x_3, x_4, x_5
        denominator = 1 + u**2 + v**2 + w**2 + t**2 + s**2
        x_1 = u * 2 / denominator
        x_2 = v * 2 / denominator
        x_3 = w * 2 / denominator
        x_4 = t * 2 / denominator
        x_5 = s * 2 / denominator
        x_6 = 1 - 2 / denominator  # This acts as the embedded coordinate into R6

        # Stack these coordinates to form the 6D points
        data_in_R6 = np.stack((x_1, x_2, x_3, x_4, x_5, x_6), axis=1)
        # Trivially embed this into R5 by adding a zero coordinate
        # data_r5 = np.hstack((data_s3, np.zeros((size, 1))))
        # data_r6 = np.hstack((data_r5, np.zeros((size, 1))))
        data=data_in_R6
        
    elif data == "trivial-s2inr6":
        theta = 2 * np.pi * np.random.random(size)
        phi =  np.pi * np.random.random(size)
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data = np.stack((x1, x2, x3), 1)
        
        # Stack these coordinates to form the 4D points
        data_s2 = np.stack((x1, x2, x3), axis=1)
        
        # Trivially embed this into R5 by adding a zero coordinate
        data_r4 = np.hstack((data_s2, np.zeros((size, 1))))
        data_r5 = np.hstack((data_r4, np.zeros((size, 1))))
        data_r6 = np.hstack((data_r5, np.zeros((size, 1))))
        data=data_r6

    elif data == "trivial-s2inr4":
        theta = 2 * np.pi * np.random.random(size)
        phi =  np.pi * np.random.random(size)
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data = np.stack((x1, x2, x3), 1)
        
        # Stack these coordinates to form the 4D points
        data_s2 = np.stack((x1, x2, x3), axis=1)
        
        # Trivially embed this into R5 by adding a zero coordinate
        data_r4 = np.hstack((data_s2, np.zeros((size, 1))))
        data=data_r4

    elif data == "randomized-s2inr4":
        theta = 2 * np.pi * np.random.random(size)  # Random azimuthal angle
        phi = np.pi * np.random.random(size)        # Random polar angle
    
        # Generate the 3D coordinates of the sphere (2-sphere in 3D space)
        x1 = np.cos(theta) * np.sin(phi)
        x2 = np.sin(theta) * np.sin(phi)
        x3 = np.cos(phi)
    
        # Stack these coordinates to form the initial 3D points
        data_s2 = np.stack((x1, x2, x3), axis=1)
    
        # Add a small random component to the fourth dimension
        x4 = 0.02 * np.random.randn(size)  # Random noise with a small standard deviation
    
        # Stack the 4D coordinates to form a sphere in R4 with some randomness
        data_r4 = np.hstack((data_s2, x4[:, None]))
        data = data_r4


    elif data == "fuzzy-line-in-r4":
        # Create a line in 2D with some noise (fuzziness)
        t = np.linspace(-1, 1, size)  # Line parameter for x1
    
        # Define the 2D line with some small noise in the second dimension
        x1 = t  # Linear component in the first dimension
        x2 = 0.1 * np.random.randn(size)  # Small fuzziness in the second dimension
    
        # Stack these 2D coordinates
        data_2d = np.stack((x1, x2), axis=1)
    
        # Embed the 2D line into R4 by adding two zero coordinates
        data_r4 = np.hstack((data_2d, np.zeros((size, 2))))  # Adding two zero coordinates to each point
    
        data = data_r4
 
    elif data == "4d-fuzzy-line-in-r4":
        # Create a linear structure in R4
        t = np.linspace(-1, 1, size)  # Line parameter (you can adjust the range as needed)
        
        # Define the line direction in R4 (you can choose different values to change the line's direction)
        x1 = t  # Linear component in the first dimension
        x2 = 0.1 * np.random.randn(size)  # Small fuzziness in the second dimension
        x3 = 0.1 * np.random.randn(size)  # Small fuzziness in the third dimension
        x4 = 0.1 * np.random.randn(size)  # Small fuzziness in the fourth dimension
        
        # Stack these coordinates to form the 4D points
        data_line_r4 = np.stack((x1, x2, x3, x4), axis=1)
        data = data_line_r4
       
 
    elif data == "hyperboloid":
        # v =2 * np.random.rand(size) -1
        v = np.linspace(-0.75,0.75,size)
        theta =2 * np.pi * np.random.rand(size)
        # v = np.linspace(-1.5,1.5,size)
        # theta =2 * np.pi * np.linspace(0,1,size)
        # v,theta=np.meshgrid(v,theta)
        # this is for making a surface otherwise some rand is needed to span the space apropriately, not with many points is very hard
        x1 = np.cosh(v)*np.cos(theta)
        x2 = np.cosh(v)*np.sin(theta)
        x3 = np.sinh(v)
        data = np.stack((x1, x2, x3), 1)
        
    
        
    elif data == "torus":
        R=1
        r=0.1
        theta = 2 * np.pi * np.linspace(0,1,size)#np.random.rand(size)
        phi = 2 * np.pi * np.random.rand(size)
        x1 = (R+r*np.cos(theta))*np.cos(phi)
        x2 = (R+r*np.cos(theta))*np.sin(phi)
        x3 = r*np.sin(theta)
        data = np.stack((x1, x2, x3), 1)
        
    elif data == "moebius":
        R=1
        w=0.2
        n=1
        # v = np.linspace(-w,w,size)
        v = w * np.random.rand(size) - w/2.
        theta = 2 * np.pi * np.random.rand(size)
        # theta = np.linspace(0,2*np.pi,size)
        # phi = 2 * np.pi * np.random.rand(size)
        x1 = (R+(v/2)*np.cos(n*theta/2))*np.cos(theta)
        x2 = (R+(v/2)*np.cos(n*theta/2))*np.sin(theta)
        x3 = (v/2)*np.sin(n*theta/2)
        data = np.stack((x1, x2, x3), 1)

    elif data == "sin-wave-mixture":
        theta_1 = 1.5*np.random.normal(size=size) - 3*np.pi/2
        theta_2 = 1.5*np.random.normal(size=size) + np.pi/2
        mixture_index = np.random.rand(size) < 0.5

        x1 = mixture_index*theta_1 + ~mixture_index*theta_2
        x2 = np.sin(x1)
        data = np.stack((x1, x2), 1)

    else:
        assert False, f"Unknown dataset `{data}''"

    return torch.tensor(data, dtype=torch.get_default_dtype())


def get_2d_datasets(name):
    train_dset = SupervisedDataset(name, "train", get_2d_data(name, size=10000))
    valid_dset = SupervisedDataset(name, "valid", get_2d_data(name, size=1000))
    test_dset = SupervisedDataset(name, "test", get_2d_data(name, size=5000))
    return train_dset, valid_dset, test_dset

