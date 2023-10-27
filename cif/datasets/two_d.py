import numpy as np

import torch

from scipy.stats import vonmises

import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

from .supervised_dataset import SupervisedDataset


# Modified from https://github.com/jhjacobsen/invertible-resnet/blob/278faffe7bf25cd7488f8cd49bf5c90a1a82fc0c/models/toy_data.py#L8 
def get_2d_data(data, size):
    if data == "swissroll":
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

    elif data == "line":
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

    elif data == "3d-von-mises-circle":
        theta = vonmises.rvs(1, size=size, loc=np.pi/2)
        phi = vonmises.rvs(1, size=size, loc=np.pi/2)/2
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data = np.stack((x1, x2, x3), 1)
        
    elif data == "3d-circle":
        theta = 2 * np.pi * np.random.random(size)
        phi =  np.pi * np.random.random(size)
        x1 = np.cos(theta)*np.sin(phi)
        x2 = np.sin(theta)*np.sin(phi)
        x3 = np.cos(phi)
        data = np.stack((x1, x2, x3), 1)


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
