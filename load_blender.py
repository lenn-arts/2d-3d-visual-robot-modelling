import os
import tensorflow as tf
import numpy as np
import imageio 
import json
import gc



# functions to generate translation and rotation matrices for camera-to-world matrix
trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)
### MANUAL for testing
trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0.5],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

# roll about x-axis
rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

# pitch about y-axis
rot_theta = lambda th : tf.convert_to_tensor([ 
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)
rot_theta_correct = lambda th : tf.convert_to_tensor([ 
    [tf.cos(th),0,tf.sin(th),0],
    [0,1,0,0],
    [-tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)

# yaw about z-axis
rot_yaw = lambda gamma : tf.convert_to_tensor([ 
    [tf.cos(gamma),tf.sin(gamma),0,0],
    [-tf.sin(gamma),tf.cos(gamma),0,0],
    [0,0,1,0],
    [0,0,0,1],
], dtype=tf.float32)
rot_yaw_correct = lambda gamma : tf.convert_to_tensor([ 
    [tf.cos(gamma),-tf.sin(gamma),0,0],
    [tf.sin(gamma),tf.cos(gamma),0,0],
    [0,0,1,0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
    

# main function
def load_blender_data(basedir, half_res=False, testskip=1, 
                      return_angles = False, remap_base=True):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_joint_angles = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        joint_angles = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
        for i_frame, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(np.array(imageio.imread(fname), dtype=np.float16)/255.)
            joints_frame = np.array(frame['joint_angle'])
            if len(joints_frame.shape) == 0: joints_frame = joints_frame.reshape((1,)) # if only one joint
            
            pose = np.array(frame['transform_matrix'])
            print("pose before\n", pose)
            if remap_base:
                pose = rot_yaw(joints_frame[0]) @ pose # new: base rotation = pose rotation
                print("pose after\n", pose)
                joints_frame[0] = 0 # reset because information was already modeled as / used in pose
            poses.append(pose)
            joint_angles.append(joints_frame)
            print("joints_frame", joints_frame)
            print(f"{i_frame}/{len(meta['frames'])}")
        
        
        #imgs_new = np.array(imgs, dtype=np.float16, copy=False)#.astype(np.float32)  # keep all 4 channels (RGBA)
        gc.collect()
        print("loop done")
        #imgs = imgs_new / 255
        imgs = np.stack(imgs, axis=0)
        print("up until imgs")
        poses = np.array(poses).astype(np.float32)
        print("up until poses")
        counts.append(counts[-1] + imgs.shape[0])
        print("up until count")
        joint_angles = (np.array(joint_angles)/np.pi).astype(np.float32) # normalize degrees in rad from rad([-180°, 180°]) to [-1,1]
        print("up until joint_angles")
        if len(joint_angles.shape) < 2: joint_angles = joint_angles.reshape(*joint_angles.shape, 1) # if only one joint
        print("joint_angles:", joint_angles.shape)

        all_imgs.append(imgs)
        all_poses.append(poses)
        all_joint_angles.append(joint_angles)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    joint_angles = np.concatenate(all_joint_angles, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x']) # fov
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = tf.stack([pose_spherical(cangle, -30.0, 4.0) for cangle in np.linspace(-180,180,40+1)[:-1]],0)
    ### MANUAL for testing: (this can be overwritten by specifying manual test poses in the test transforms and setting render_test = True in the config)
    render_poses = tf.stack([pose_spherical(cangle, 0, 2.25) for cangle in np.linspace(-180,180,25+1)[:-1]],0) # horizontal
    #render_poses = tf.stack([pose_spherical(0, cangle, 2.25) for cangle in np.linspace(-180,180,20+1)[:-1]],0)

    # new 2023-04-04
    num_render_angles = 1
    render_joint_angles = tf.stack([[0,j_angle, j_angle] for j_angle in np.linspace(-2.7,2.7,25)] , 0)
    #render_joint_angles = tf.stack([[0,j_angle] for j_angle in np.linspace(0,0,20)] , 0) # for video
    #render_joint_angles = tf.stack([[j_angle]*num_render_angles for j_angle in np.linspace(0,0,20)] , 0) # new for base
    render_joint_angles /= np.pi
    print("load_blender_data > render_joint_angles", render_joint_angles.shape, render_joint_angles)

    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.
    gc.collect()
    if return_angles:
        return imgs, poses, render_poses, [H, W, focal], i_split, joint_angles, render_joint_angles
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split


