import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# X) Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dim = 3):

    if input_dim == 3:
        if i == -1:
            return tf.identity, 3 # 3 is only accurate 

        embed_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires-1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [tf.math.sin, tf.math.cos],
        }
    # NEW 2023-04-10 FOR JOINT ANGLES
    else:
        if i == -1: return tf.identity, input_dim

        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dim,
            'max_freq_log2': multires-1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [tf.math.sin, tf.math.cos],
        }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim # returns function + how many dimensions in the input to the model belong this embedding (e.g. 3->63 for point coordinates)


# EXP: THIS IS THE NEURAL NETWORK USED
# X) Model architecture

# called by: create_nerf()
# DETAILS: points are points in 3d space, usually sampled from one ray (one straight line); directions are the rays?
# ARGS: D = depth (number of MLP layers); W = number of units per layer; input_ch=3 (each point is 3d coordinate vector)
def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, input_ch_joint_angles=1, output_ch=4, skips=[4], use_viewdirs=False):

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act) # returns a fully connected layer, which itself acts like a function

    print('init_nerf_model > MODEL created: input_ch, input_ch_views, input_ch_joint_angles, use_viewdirs', 
          input_ch, input_ch_views, input_ch_joint_angles, use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)
    input_ch_joint_angles = int(input_ch_joint_angles) 

    inputs = tf.keras.Input(shape=(input_ch_joint_angles + input_ch + input_ch_views))
    inputs_pts, inputs_views, input_angles = tf.split(inputs, [input_ch, input_ch_views, input_ch_joint_angles], -1) # split into two tensors of sizes ? X input_ch and ? X input_ch_views (along last dimension of tensor)
    input_angles.set_shape([None, input_ch_joint_angles])
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    # new NN that maps base angle and coordinates (both positionally encoded) to new coordinates
    pre_outputs = tf.concat([inputs_pts,input_angles], -1)
    for i in range(4):
        pre_outputs = dense(256)(pre_outputs)
    #pre_outputs = dense(128)(pre_outputs)
    pre_outputs = dense(input_ch)(pre_outputs) # new coordinates (no activation since regression)

    print("init_nerf_model > inputs.shape, inputs_pts.shape, inputs_views.shape", inputs.shape, inputs_pts.shape, inputs_views.shape)
    #outputs = tf.concat([input_angles, inputs_pts], -1) # old
    outputs = tf.concat([inputs_pts+pre_outputs], -1) # new NN
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips: # skip connection (concatenate original input with outputs of this layer as input to next layer)
            outputs = tf.concat([input_angles, inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)  # THIS IS THE OCCUPANCY (OUTPUT LAYER 1)
        bottleneck = dense(256, act=None)(outputs) # this is a parellel layer at the same depth as where the occupancy gets generated
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs  # input (bottleneck from 8 prev layers + now viewing directions) into the last layer that predicts color
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1): # one extra hidden layer
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs) # 3 because three output values (R,G,B) (OUTPUT LAYER 2)
        # even though no activation is used here, the rgb values are sigmoided, and alpha/density is further processed in run_nerf.py > render_rays()
        outputs = tf.concat([outputs, alpha_out], -1) # concat RGB with Occupancy
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# X) Ray helpers

# called by: render(); train()
def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    # normalization so that coordinate (50,50) is (0,0) in a W=100,H=100 image -->
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1) # stack together tensors of shape iX1, jX1, iX1
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # multiply direction tensors (normalized to the ray origin/pinhole point) elementwise with rotation of pose relative to the world coordinate space
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d)) # origin is just the last column of c2w matrix, i.e. the translation of the pose relativ to the world coordinate space origin
    return rays_o, rays_d # 3x1, 3x1

# called by: train()
def get_rays_np(H, W, focal, c2w): # np=numpy
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    # normalization so that coordinate (50,50) is (0,0) in a W=100,H=100 image -->
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # per dir: elementwise multiplication between dir and c2w
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d  # shape: HxWx3 ; HxWx3

# called by: render()
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# X) Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
