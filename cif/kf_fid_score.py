# from https://github.com/bioinf-jku/TTUR/blob/master/fid.py

import os
import numpy as np
import tensorflow.compat.v1 as tf
from scipy import linalg
from tensorflow.compat.v2.keras.utils import get_file
import tarfile

def is_local():
    # import socket
    # return 'mtv' in socket.gethostname()
    return True

class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# -------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    # layername = 'FID_Inception_Net/pool_3:0'
    layername = 'pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                # shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


# -------------------------------------------------------------------------------

def get_activations(images, sess, batch_size=50, verbose=False):
    # print("*** get_activations")
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        # pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred = sess.run(inception_layer, {'ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


# -------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print('warn:' + msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=128, verbose=False):
    # print("*** calculate_activation_statistics")
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
# -------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    download_file = inception_path + '/classify_image_graph_def.tar'
    model_file = inception_path + '/classify_image_graph_def.pb'
    if is_local():
        if not os.path.exists(model_file):
            print("Downloading Inception model")

            download_path = get_file(download_file, origin=INCEPTION_URL)

            with tf.gfile.Open(download_path, mode='rb') as gf:
                with tarfile.open(fileobj=gf, mode='r') as f:
                    f.extract('classify_image_graph_def.pb', inception_path)
        return str(model_file)
    else:
        return '/cns/iz-d/home/nijkamp/models/inception/classify_image_graph_def.pb'


def fid_from_samples(x1, x2, init=True):
  if (x1.ndim==3) | (x1.shape[3]==1):
    x1 = np.concatenate([x1, x1, x1], axis=3)
    #print("Added channel dimension.")
  if (x2.ndim==3) | (x2.shape[3]==1):
    x2 = np.concatenate([x2, x2, x2], axis=3)
    #print("Added channel dimension.")
  inception_path = None
  if init: inception_path = check_or_download_inception(None)
  if init: create_inception_graph(str(inception_path))
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      #print("* calculate_fid_given_paths")
      m1, s1 = calculate_activation_statistics(x1, sess)
      #print("** calculate_fid_given_paths")
      m2, s2 = calculate_activation_statistics(x2, sess)
      #print("*** calculate_fid_given_paths")
      fid_value = calculate_frechet_distance(m1, s1, m2, s2)
      return fid_value
      
def m_s_from_samples(x):
  if x.shape[3]==1:
    x = np.concatenate([x, x, x], axis=3)
    #print("Added channel dimension.")
  inception_path = check_or_download_inception(None)
  create_inception_graph(str(inception_path))
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      #print("* calculate_fid_given_paths")
      m, s = calculate_activation_statistics(x, sess)
      #print("** calculate_fid_given_paths")
      return m, s
      
#x1 = np.rollaxis(iter(train_loader).next()[0].numpy(), 1, 4)
#print("*** ", x1.shape)
#x2 = np.rollaxis(iter(test_loader).next()[0].numpy(), 1, 4)
#print("*** ", x2.shape)
#fid = fid_from_samples(x1, x2)
#print(fid)
