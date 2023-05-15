import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false' # manually deactivated

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from run_nerf_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from numba import cuda
import gc

gc.enable()
gc.collect()

###os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.compat.v1.enable_eager_execution()
#tf.keras.backend.clear_session() # MANUAL
#cuda.get_current_device()
#cuda.close()
#device = cuda.get_current_device()
#device.reset()

# called by: run_network()
# calls: fn-> 
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

# EXP: main forward pass of the network
# called by: network_query_fn()
# calls: embed_fn->; batchify()
def run_network(inputs, viewdirs, joint_angles, fn, embed_fn, embeddirs_fn, embedjoints_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
        args:
        joint_angles: [N_rays_of_batch(N_rand) X num_joints]"""

    ###print("run_network > inputs.shape", inputs.shape)
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]]) # (N_rays_of_batch*N_points_on_ray) X 3   # 65536 X 3
    
    ###print("run_network > inputs_flat.shape", inputs_flat.shape)
    embedded = embed_fn(inputs_flat) # from 3 coordinates per point to 63 # (N_rays_of_batch*N_points_on_ray) X 63
    ###print("run_network > embedded.shape", embedded.shape)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1) # (N_rays_of_batch*N_points_on_ray) X (90)
        ###print("run_network > embedded.shape after tf.concat([embedded, embedded_dirs], -1)", embedded.shape)

    # NEW 2023-03-28
    #joint_angles = tf.broadcast_to(joint_angles[:, None, None], inputs.shape[0:2]+[1]) # insert 2nd and 3rd dimension and broadcast to N_rays_of_batch X N_points_on_ray X 1
    ###print("run_network > joint_angles.shape", joint_angles.shape)
    joint_angles = tf.broadcast_to(joint_angles[:, None, :], (joint_angles.shape[0],inputs.shape[1],joint_angles.shape[-1])) # insert 2nd dimension and broadcast to N_rays_of_batch X N_points_on_ray* X num_joints
    joing_angles_flat = tf.reshape(joint_angles, [-1, joint_angles.shape[-1]]) # reshape to (N_rays_of_batch*N_points_on_ray) X num_joints
    ###print("run_network > joint_angles.shape", joint_angles.shape)
    joing_angles_flat = tf.cast(joing_angles_flat, tf.float32)
    ###print("run_network > joing_angles_flat.shape", joing_angles_flat.shape)
    embedded_joint_angles = embedjoints_fn(joing_angles_flat)
    ###print("run_network > joint_angles_flat_embed.shape", embedded_joint_angles.shape)

    embedded = tf.concat([embedded, embedded_joint_angles], -1)  # [embedded_inputs, embedded_views, embedded_angles]
    ###print("run_network > embedded.shape after tf.concat([embedded, joint_angles_flat], -1)", embedded.shape)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

### called by: batchify_rays()
### calls: network_query_fn()
def render_rays(ray_batch,
                joint_angles,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    """Volumetric rendering.

    Args:
      ray_batch:  array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn:  function. Model for predicting RGB and density at each point
        in space.
      network_query_fn:  function used for passing queries to network_fn.
      N_samples:  int. Number of different times to sample along each ray.
      retraw:  bool. If True, include model's raw, unprocessed predictions.
      lindisp:  bool. If True, sample linearly in inverse depth rather than in depth.
      perturb:  float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance:  int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine:  "fine" network with same spec as network_fn.
      white_bkgd:  bool. If True, assume a white background.
      raw_noise_std:  ...
      verbose:  bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model. -> as in: for each point (=sample) 4 values are predicted (alpha + RGB)
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model. -> se above
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): 
            return 1.0 - tf.exp(-act_fn(raw) * dists) # relu to cut negative values; also see formula for quadrature in paper

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1] # last n-1 values - first n-1 values

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3] # sigmoid to bring into range (0.0,1.0)

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # based on (4th-output + noise) and distances # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True) #tf.math.cumprod([a, b, c]) = [a, a * b, a * b * c]

        # Computed weighted color of each sample along each ray. -> done as weighted sum of sampled points' rgb values
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance. -> expected distance that the ray travels before reaching a barrier?
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction from ray_batch. (first 6)
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction from ray_batch. (last 3)
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None # [N_rays, 3]  # not (but almost) the same as rays_d

    # Extract lower, upper bound for ray distance. (7,8)
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples]) # [N_rays, N_samples]

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    print("render_rays > ray_batch.shape", ray_batch.shape)
    print("render_rays > pts.shape", pts.shape)
    print("render_rays > joint_angles.shape", joint_angles.shape)

    # EXP: MAIN CALL TO NETWORK!
    # Evaluate model at each point.
    #raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    raw = network_query_fn(pts, viewdirs, joint_angles, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # EXP: MAIN CALL TO NETWORK! 2
        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, joint_angles, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    # return object
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map} 
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret

### called by: render()
### calls: render_rays()
def batchify_rays(rays_flat, joint_angles, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    print("batchify_rays > rays_flat.shape", rays_flat.shape)
    print("batchify_rays > joint_angles.shape", joint_angles.shape)
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], joint_angles[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret

### called by: render_path(), train()>update routine, and train()>log image
### calls: get_rays(), batchify_rays()
def render(H, W, focal,
           chunk=1024*32, rays=None, joint_angles=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      joint_angles: either of shape [N_poses_of_batch*H*W, num_joints] when rays are given and c2w is not (direct call from train) 
                    OR of shape [N_poses, num_joints] when c2w is given and rays are not (indirect from train via render_path())
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        print("render > rays_o.shape, rays_d.shape", rays_o.shape, rays_d.shape)
        print("render > joint_angles.shape", joint_angles.shape)
        joint_angles = get_joint_angles_per_ray(joint_angles,W=W,H=H)
        print("render > after batchify: joint_angles.shape", joint_angles.shape)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray -> this goes into render_rays() !
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    print("render > rays.shape", rays.shape)
    all_ret = batchify_rays(rays, joint_angles, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

# mostly used for test/val image rendering, not for trianing
def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, joint_angles=None):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, focal, joint_angles=np.array([joint_angles[i]]), chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

# EXP: creates the model (fine and coarse), embed_fn, the train and test render arguments, the gradient variables
# called_by: train()
def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)


    # NEW 2023-04-04
    ##input_ch_joint_angles = 1
    embedjoints_fn, input_ch_joint_angles = get_embedder(
            args.multires_joints, args.i_embed, input_dim=args.num_joints)

    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
        input_ch_joint_angles=input_ch_joint_angles)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
            input_ch_joint_angles=input_ch_joint_angles)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    ### calls run_network
    ### called by render_rays>raw2output
    ### args: inputs are 3d points to evaluate (occupancy+color), viewdirs are directions of rays (vectors) to eval. color , network_fn is function-arg in run_network
    def network_query_fn(inputs, viewdirs, angles, network_fn): return run_network(
        inputs, viewdirs, angles, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        embedjoints_fn=embedjoints_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # render args for test/val images are almost same as for train
    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    ### CHANGED: added this line
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of training iterations')
    
    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D (EXP: use direction of viewing as predictor for color OR just predict color like occupancy from position only)')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    # NEW 2023-04-10
    parser.add_argument("--multires_joints", type=int, default=10,
                        help='log2 of max freq for positional encoding of joint angles (1 angle = 1 float)')
    parser.add_argument("--num_joints", type=int, default=1,
                        help='Will be overridden later by automatic inference of the number of joints from input transforms.json')
    parser.add_argument("--remap_base_angle", type=int, default=1,
                        help="use base joint angle to rotate the c2w transform matrix")
    
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=50,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=100,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=200,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=500,
                        help='frequency of render_poses video saving')

    return parser

def get_joint_angles_per_ray(joint_angles_per_pose, rays=None, no_imgs=0, W=None, H=None, keep_dims=False):
    """ given rays or img dimensions, transform joint angles per pose to joint angle per ray (each image has H*W rays)
        ARGS::
        joint_angles:   [N_imgs X num_joints]   (even for N_imgs=1)
        rays:           [(N_imgs*H*W) X ro+rd+rgb (3) X 3]
        RETURN::
        joint_angle_per_ray:    [[(N_imgs*H*W) X num_joints] OR [H X W X num_joints]"""
    if (no_imgs > 0) and (rays is not None):
        joint_angle_per_ray = np.repeat(joint_angles_per_pose, rays.shape[0]/no_imgs, axis=0)
    elif (W is not None) and (H is not None):
        if keep_dims is False:
            joint_angle_per_ray = np.repeat(joint_angles_per_pose, W*H, axis=0)
        else:
            joint_angles_per_pose = joint_angles_per_pose[:, np.newaxis, np.newaxis, ...]
            joint_angle_per_ray = np.repeat(joint_angles_per_pose, H, axis=1)
            joint_angle_per_ray = np.repeat(joint_angle_per_ray, W, axis=2)
            joint_angle_per_ray = np.squeeze(joint_angle_per_ray, axis=0)
    else:
        raise Exception("too little args to compute joint_angles_per_ray")
    return joint_angle_per_ray

def train():

    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # 1) Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # poses are all my poses (train, val, test)
        # render_poses are artificially generated poses 
        args.remap_base_angle = True if args.remap_base_angle == 1 else False
        images, poses, render_poses, hwf, i_split, joint_angles, render_joint_angles = load_blender_data(
            args.datadir, args.half_res, args.testskip, return_angles = True, remap_base=args.remap_base_angle)
        print('train > Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        print("train > joint_angles", joint_angles.shape, joint_angles[:10])
        args.num_joints = joint_angles.shape[-1]
        i_train, i_val, i_test = i_split
        # i_train etc are arrays of indices used to select from all poses to retrieve train set, val set etc

        near = 0.1#0.1#2.#0.5#2#0.1#2.
        far = 6.#3.#6.#2.5#6#99#6.
        print("******* NEAR, FAR:", near, far, "*******")

        if args.white_bkgd: 
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    if args.render_test: ### render user-given test poses if set to true (instead of generated ones)
        render_poses = np.array(poses[i_test])
        render_joint_angles = np.array(joint_angles[i_test])

    # 2) Cast intrinsics to right types
    H, W, focal = hwf
    focal = 555.55#300.0#278.0#153.0#150.0 #best so far: 300 #others: 17#61.7#64#13.#1.#5.#20.#10. #10 #64 #48 #154.51 #85.333 #39 #10 #64  # <------
    print(f"******* LRATE: {args.lrate} *******")
    print(f"******* FOCAL: {focal} *******")
    print(f"******* HEIGHT: {H} *******")
    print(f"******* WIDTH: {W} *******")
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 3) Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 4) Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)
    # render arguments for train set, and test render, start?, variables for which to compute gradients, [coarse model, fine model]

    bds_dict = { # boundaries
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering test images/videos out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test: # use user-given test poses (this is actually set above)
            # render_test switches to test poses
            images = images[i_test]
        else: # use artificially generated poses
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, 
                              joint_angles=render_joint_angles)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=10, quality=8)

        return

    # 5) Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] (N total number of images) where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('train > get rays (get_rays_np())')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3] (where 3=(x,y,z)?) for each pixel in the image = [2, H, W, 3].
        # This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]] # list of (list of rays for pose) for all poses
        rays = np.stack(rays, axis=0)  # [N, ro+rd (2), H, W, 3]
        print('train > rays.shape', rays.shape)
        print('train > done, concats')
        # concatenate rays with rgb value at the pixel the ray corresponds to
        # [N, ro+rd+rgb, H, W, 3] ->
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1) 
        # [N, H, W, ro+rd+rgb, 3] -> 
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        print('train > rays_rgb.shape', rays_rgb.shape) 
        # [N_train, H, W, ro+rd+rgb, 3] -> 
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        print('train > rays_rgb.shape', rays_rgb.shape)                
        # [N_train*H*W, ro+rd+rgb (3), 3] -> this merges rays from different images in 0-th dimension -> 
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('train > rays_rgb.shape', rays_rgb.shape)
        pose_indices_per_ray = np.repeat(np.array(i_train), rays_rgb.shape[0]/len(i_train))
        joint_angle_per_ray = get_joint_angles_per_ray(joint_angles, rays=rays_rgb, no_imgs=len(i_train))
        ###joint_angle_per_ray = np.repeat(joint_angles, rays_rgb.shape[0]/len(i_train))
        print('train > pose_indices_per_ray.shape', pose_indices_per_ray.shape, pose_indices_per_ray[159998:160004])
        print('train > shuffle rays')
        rng = np.random.default_rng()
        shuffled_indexes = rng.permutation(rays_rgb.shape[0])
        print("train > shuffle", shuffled_indexes[:5], pose_indices_per_ray[:5], joint_angle_per_ray[:5])
        rays_rgb = rays_rgb[shuffled_indexes]
        pose_indices_per_ray = pose_indices_per_ray[shuffled_indexes]
        joint_angle_per_ray = joint_angle_per_ray[shuffled_indexes]
        print("train > shuffle", shuffled_indexes[:5], pose_indices_per_ray[:5], joint_angle_per_ray[:5])
        print('train > done')
        i_batch = 0

    N_iters = 1000000
    print('train() > Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # 6) Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    # 7) training
    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch

        if use_batching:
            # Random rays from all (training?) images
            # this selects from rays_rgb in the first dimension and keeps the others
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?] # randomly (via prior shuffle) select N_rand different train images
            batch = tf.transpose(batch, [1, 0, 2]) # [ro+rd+rgb, batch*H*W, 3]

            batch_joint_angles = joint_angle_per_ray[i_batch:i_batch+N_rand] # [N_rand, 1]

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2] # [ro+rd, batch*H*W, 3] + [rgb, batch*H*W, 3]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]: # wrap around
                np.random.shuffle(rays_rgb)
                i_batch = 0

        else:
            # Random from one image of the training set
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            batch_joint_angles = get_joint_angles_per_ray(joint_angles[img_i][np.newaxis,...], H=H, W=W, keep_dims=True)
            print("batch_joint_angles.shape", batch_joint_angles.shape)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose) # [H, W, 3], [H, W, 3]
                #batch_joint_angles = get_joint_angles_per_ray(joint_angles[img_i], rays=rays_o+rays_d, no_imgs=1)
                
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False) # select N_rand pixels/rays from select train img
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis]) # [N_batch, x+y(2)]
                print("rays_o.shape", rays_o.shape)
                rays_o = tf.gather_nd(rays_o, select_inds) # [N_batch, 3]
                print("rays_o.shape", rays_o.shape)
                rays_d = tf.gather_nd(rays_d, select_inds)
                print("batch_joint_angles.shape, select_inds.shape", batch_joint_angles.shape, select_inds.shape, select_inds[:5])
                batch_joint_angles = tf.gather_nd(batch_joint_angles, select_inds) # TODO: verify whether this works
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make NN predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, joint_angles=batch_joint_angles, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0

        #####           end            #####

        # 8) Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        ### EXP: render 2 train images turing training and save them
        ### CHANGED: added this block to record images
        if i%args.i_img==0 and i > 0:
            print("SHAPE: ", render_poses.shape)
            print("SHAPE new: ", render_poses[0,:,:].shape)
            imgbase = os.path.join(basedir, f"{expname}")#, '{}_spiral_{:06d}_.png'.format(expname, i))
            #rgb, disp = render_path(render_poses[0:2, :, :], hwf, args.chunk, render_kwargs_test, render_factor=1, savedir=imgbase)
            rgb, disp = render_path(poses[0:2, :, :], hwf, args.chunk, render_kwargs_test, 
                                    joint_angles=joint_angles[0:2,:], render_factor=1, savedir=imgbase)

        if i % args.i_video == 0 and i > 0:   #### save video from generate poses
            rgbs, disps = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test, joint_angles=render_joint_angles)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=10, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, disps_still = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test, joint_angles=render_joint_angles)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=10, quality=8)
                imageio.mimwrite(moviebase + 'disp_still.mp4',
                             to8b(disps_still / np.max(disps_still)), fps=10, quality=8)

        if i % args.i_testset == 0 and i > 0:  ### render given testset poses?
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir, joint_angles=joint_angles[i_test])
            print('Saved test set')

        if i % args.i_print == 0 or i < 10:  ### print updates to console

            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

            if i % args.i_img == 0:  ### save image

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)  ### RANDOM VALIDATION IMAGE
                target = images[img_i]
                pose = poses[img_i, :3, :4] # pose of validation image
                # new 2023-04-04
                min_angle, max_angle =  -1.0, 1.0#-2.9, 2.9
                #val_joint_angle = np.array([0.0]*args.num_joints) #random.sample(range(int(min_angle*100),int(max_angle*100)+1),k=1)/100.0
                # a) custom val joint angles
                val_joint_angle = np.array([0.0, 
                                            np.random.uniform(low=min_angle, high=max_angle),
                                            np.random.uniform(low=min_angle, high=max_angle)]) # for 3 joints
                # b) original joint angles
                val_joint_angle = joint_angles[img_i]
                print("train > val_joint_angle", val_joint_angle.shape)
                if val_joint_angle.ndim < 2: val_joint_angle = np.reshape(val_joint_angle, (1, val_joint_angle.shape[-1]))
                print("train > val_joint_angle after", val_joint_angle.shape)
                rgb, disp, acc, extras = render(H, W, focal, joint_angles=val_joint_angle, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test) # render pose

                psnr = mse2psnr(img2mse(rgb, target))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i==0:
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}_j{}_a{}.png'.format(i,img_i, np.round(val_joint_angle,2))), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()
