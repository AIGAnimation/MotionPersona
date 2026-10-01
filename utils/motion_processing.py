"""
This script is written to handle the motion data in BVH format, including loading, saving, and processing.
It can work with bvh_motion.py to operate the motion class in different ways.
"""

import sys
sys.path.append('./')

import numpy as np

from utils.bvh_motion import Motion
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

'''
Rotation with Quaternion (w, x, y, z)
'''

def remove_joints(motion: Motion, remove_key_words):
    remove_name, remove_idxs = [], []
    for name_idx, name in enumerate(motion.names):
        for key_word in remove_key_words:
            if key_word in name.lower():
                remove_name.append(name)
                remove_idxs.append(name_idx)
                
    # remove child joints
    for idx in range(len(motion.names)):
        if motion.parents[idx] in remove_idxs and idx not in remove_idxs:
            remove_name.append(motion.names[idx])
            remove_idxs.append(idx)
    
    ori_name, ori_parents = motion.names.copy(), motion.parents.copy()
    removed_rotation = motion.rotations[:, remove_idxs]
    removed_offset = motion.offsets[remove_idxs]
    motion.rotations = np.delete(motion.rotations, remove_idxs, axis=1)
    motion.positions = np.delete(motion.positions, remove_idxs, axis=1)
    motion.offsets = np.delete(motion.offsets, remove_idxs, axis=0)
    motion.names = np.delete(motion.names, remove_idxs, axis=0)
    
    # update parents
    motion.parents = np.zeros(len(motion.names), dtype=np.int32)
    for idx in range(len(motion.names)):
        parent_name = motion.parent_names[motion.names[idx]]
        if parent_name is None:
            motion.parents[idx] = -1
        else:
            motion.parents[idx] = np.where(motion.names == parent_name)[0][0]

    motion.names = motion.names.tolist()
    motion.removed_joints.append([removed_rotation, removed_offset, remove_idxs, ori_parents])
    motion.opt_history.append('remove_joints')
    return motion


def scaling(motion: Motion, scaling_factor=None):
    if scaling_factor is None:
        t_pose = motion.get_T_pose()
        heights = t_pose[:, 1]
        height_diff = np.max(heights) - np.min(heights)
        scaling_factor = height_diff
    motion.positions *= scaling_factor
    motion.offsets *= scaling_factor
    motion.end_offsets *= scaling_factor
    motion.scaling_factor = scaling_factor
    motion.opt_history.append('scaling')
    return motion

# This function only works for the motion where the T-pose is the -y positive direction
def mirror(motion: Motion, l_joint_idxs, r_joint_idxs):
    ori_rotations = motion.rotations.copy()
    # mirror root trajectory
    motion.positions[:, :, 0] *= -1
    
    # mirror joint rotations
    motion.rotations[:, l_joint_idxs] = ori_rotations[:, r_joint_idxs]
    motion.rotations[:, r_joint_idxs] = ori_rotations[:, l_joint_idxs]
    motion.rotations[:, :, [2, 3]] *= -1
    
    # mirror hip rotations
    motion.rotations[:, 0] *= -1 
    motion.opt_history.append('mirror')
    return motion


def on_ground(motion: Motion):
    global_pos = motion.global_positions
    lowest_height = np.min(global_pos[:, :, 1])
    motion.positions[:, :, 1] -= lowest_height
    motion.opt_history.append('on_ground')
    return motion


def root(motion: Motion, given_pos=None, return_pos=False):
    root_init_pos = motion.positions[0, 0, [0, 2]] if given_pos is None else given_pos
    motion.positions[:, 0, [0, 2]] -= root_init_pos
    motion.opt_history.append('root')
    if return_pos:
        return motion, root_init_pos
    else:
        return motion


def temporal_scale(motion: Motion, scale_factor: int):
    motion.positions = motion.positions[::scale_factor]
    motion.rotations = motion.rotations[::scale_factor]
    motion.frametime *= scale_factor
    motion.opt_history.append('temporal_scale')
    return motion


def extract_contact_adaptive(motion, l_foot_idx, r_foot_idx, vel_threshold=0.2, height_threshold=5.0,
                             smooth_sigma=1.5, ref_window=121, min_density=0.15,
                             relax=(1.5, 2.25)):
    """Contact labels robust to two failure modes of the raw captures:

    * marker jitter: stance-foot speed never dips under the threshold on noisy rigs (some performers have
      stance p20 speeds of 0.12-0.19 cm/frame vs 0.03-0.06 on clean rigs) -> the foot trajectory is
      gaussian-smoothed (sigma ~1.5 frames at 30 fps) before the speed test;
    * within-clip height drift: the height guard was relative to the clip-global minimum, which a drifting
      baseline pushes 6-8 cm away -> the reference is a rolling minimum over ``ref_window`` frames (~4 s).

    If a foot still labels fewer than ``min_density`` of frames, the velocity threshold is relaxed by the
    ``relax`` factors (capped at 0.45 cm/frame = 0.135 m/s by default; calibrated on the own-body captures -
    this keeps the labelled-frame raw-speed p90 within ~0.3 cm/frame, i.e. fixed-threshold precision, while
    doubling density on the jittery subjects; a 0.25 floor with a 1 cm/frame cap over-labels) until the
    density is reached - genuinely airborne clips simply keep the last attempt (~2 %).
    Returns (T, 2) float 0/1 like ``extract_contact``.
    """
    from scipy.ndimage import minimum_filter1d
    contact = np.zeros((motion.frame_num, 2))
    for col, idx in ((0, l_foot_idx), (1, r_foot_idx)):
        pos = gaussian_filter1d(motion.global_positions[:, idx].astype(np.float64), smooth_sigma, axis=0, mode='nearest')
        vel = np.sqrt((np.diff(pos, axis=0) ** 2).sum(-1))
        vel = np.append(vel, vel[-1])
        h = pos[:, 1] - minimum_filter1d(pos[:, 1], size=int(ref_window), mode='nearest')
        c = (vel <= vel_threshold) & (h <= height_threshold)
        for k in relax:
            if c.mean() >= min_density:
                break
            c = (vel <= vel_threshold * k) & (h <= height_threshold)
        contact[:, col] = c
    return contact


def extract_contact(motion, l_foot_idx, r_foot_idx, vel_threshold=0.05, height_threshold=3):
    contact_labels = np.zeros((motion.frame_num, 2))  # 0 for off-contact, 1 for on-contact
    
    left_foot_pos = motion.global_positions[:, l_foot_idx].copy()
    right_foot_pos = motion.global_positions[:, r_foot_idx].copy()
    
    left_foot_vel = np.sqrt(np.sum(np.diff(left_foot_pos, axis=0)**2, axis=1))
    right_foot_vel = np.sqrt(np.sum(np.diff(right_foot_pos, axis=0)**2, axis=1))
    left_foot_vel = np.append(left_foot_vel, left_foot_vel[-1])
    right_foot_vel = np.append(right_foot_vel, right_foot_vel[-1])
    
    left_foot_height = left_foot_pos[:, 1] - np.min(left_foot_pos[:, 1])
    right_foot_height = right_foot_pos[:, 1] - np.min(right_foot_pos[:, 1])
    
    contact_labels[:, 0] = (left_foot_vel <= vel_threshold) & (left_foot_height <= height_threshold)
    contact_labels[:, 1] = (right_foot_vel <= vel_threshold) & (right_foot_height <= height_threshold)
    
    contact_labels = contact_labels.astype(int)
    
    return contact_labels

'''
frame_idx: int or list
'''
def extract_forward(motion: Motion, frame_idx, left_shoulder_name, right_shoulder_name, left_hip_name, right_hip_name, return_forward=False):
    if type(frame_idx) is int:
        frame_idx = [frame_idx]
    
    names = list(motion.names)
    try:
        l_s_idx, r_s_idx = names.index(left_shoulder_name), names.index(right_shoulder_name)
        l_h_idx, r_h_idx = names.index(left_hip_name), names.index(right_hip_name)
    except:
        raise Exception('Cannot find joint names, please check the names of Hips and Shoulders in the bvh file.')
    global_pos = motion.update_global_positions()
    
    upper_across = global_pos[frame_idx, l_s_idx, :] - global_pos[frame_idx, r_s_idx, :]
    lower_across = global_pos[frame_idx, l_h_idx, :] - global_pos[frame_idx, r_h_idx, :]
    across = upper_across / np.sqrt((upper_across**2).sum(axis=-1))[...,np.newaxis] + lower_across / np.sqrt((lower_across**2).sum(axis=-1))[...,np.newaxis]
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward_angle = np.arctan2(forward[:, 2], forward[:, 0])
    if return_forward:
        return forward_angle, forward
    return forward_angle


'''
frame_idx: int or list
'''
def extract_forward_hips(motion: Motion, frame_idx, left_hip_name, right_hip_name, return_forward=False):
    if type(frame_idx) is int:
        frame_idx = [frame_idx]
    
    names = list(motion.names)
    try:
        l_h_idx, r_h_idx = names.index(left_hip_name), names.index(right_hip_name)
    except:
        raise Exception('Cannot find joint names, please check the names of Hips and Shoulders in the bvh file.')
    global_pos = motion.update_global_positions()
    
    lower_across = global_pos[frame_idx, l_h_idx, :] - global_pos[frame_idx, r_h_idx, :]
    across = lower_across / np.sqrt((lower_across**2).sum(axis=-1))[...,np.newaxis]
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward_angle = np.arctan2(forward[:, 2], forward[:, 0])
    if return_forward:
        return forward_angle, forward
    return forward_angle


def extract_path_forward(motion: Motion, start_frame=0, end_frame=60):
    if motion.frame_num < end_frame:
        end_frame = motion.frame_num - 1
    root_xz = motion.positions[:, 0, [0, 2]]
    root_xz_offset = root_xz[end_frame] - root_xz[start_frame]
    forward_angle = np.arctan2(root_xz_offset[1], root_xz_offset[0])
    return forward_angle 


def rotate(motion: Motion, given_angle=None, axis='y', return_angle=False):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if len(given_angle) > 1:
        assert len(given_angle) == motion.frame_num
        rot_vec = np.zeros((motion.frame_num, 3))
        rot_vec[:, axis_map[axis]] = given_angle
        given_rotation = R.from_rotvec(rot_vec)
    else:
        rot_vec = np.zeros(3)
        rot_vec[axis_map[axis]] = given_angle
        given_rotation = R.from_rotvec(rot_vec)
    ori_root_R_xyzw = motion.rotations[:, 0][..., [1, 2, 3, 0]]
    ori_root_pos = motion.positions[:, 0]
    rotated_root = (given_rotation * R.from_quat(ori_root_R_xyzw)).as_quat()[..., [3, 0, 1, 2]]
    rotated_root_pos = given_rotation.apply(ori_root_pos)
    motion.rotations[:, 0] = rotated_root
    motion.positions[:, 0] = rotated_root_pos
    motion.opt_history.append('rotate_%s_%s' % (axis, given_angle))
    return motion


def dead_blending(prev_rot, next_rot, wnd_size=15):
    '''
    Dead blending for animation clips
    prev_rot: the first animation sequence of quaternions
    next_rot: the second animation sequence
    wnd_size: length of corss-fade blend frames
    '''
    # Q: How to make the smooth transition between two clips?
    # ref to: https://theorangeduck.com/page/dead-blending 
    updated_next_rot = next_rot.copy()
    wnd_size = min(wnd_size, updated_next_rot.shape[0])
    rot0 = R.from_quat(prev_rot[-2])
    rot1 = R.from_quat(prev_rot[-1])
    delta = rot1 * (rot0.inv())
    ext_rot = delta * rot1

    for i in range(0, wnd_size):
        alpha = (i+1)/wnd_size
        ext_rot_quat = ext_rot.as_quat()
        for rid in range(0, updated_next_rot[i].shape[0]):
            if np.dot(ext_rot_quat[rid], updated_next_rot[i][rid]) < 0.0:
                updated_next_rot[i][rid] *= -1 # resolve the quaternion double cover: keep the shortest-arc rotation
        updated_next_rot[i] = ext_rot_quat * (1-alpha) + updated_next_rot[i] * alpha
        updated_next_rot[i] /= np.sqrt(np.sum(updated_next_rot[i]**2, axis=1)).reshape(updated_next_rot[i].shape[0], 1)
        ext_rot = delta * ext_rot
    
    return updated_next_rot


def norm_vec(vec):
    norm = np.sqrt(np.sum(vec ** 2))
    if norm == 0.0:
        return np.zeros_like(vec)
    else:
        return vec / norm

def from_to_rotation(from_vec, to_vec):
    '''
    returns the quaternion needed to rotate from_vec to to_vec
    '''
    from_vec = norm_vec(from_vec)
    to_vec = norm_vec(to_vec)
    cos_theta = np.dot(from_vec, to_vec)
    if np.abs(cos_theta-1.0) < 1e-5:
        return R.from_quat([0, 0, 0, 1])
    theta = np.arccos(cos_theta)
    axis = norm_vec(np.cross(from_vec, to_vec))
    quat = np.zeros((4))
    quat[0:3] = np.sin(theta/2)*axis # x,y,z
    quat[3] = np.cos(theta/2) # w
    return R.from_quat(quat)

def fabrik(arm_length, parents, chain, target_pos, rotation_ref, positions, max_iter=20):
    '''
    Reach the target joint (target_ind) to target_pos, adjust rotations of related joints
    chain[ 0] -> target joint
    chain[-1] -> root joint
    '''
    # keep a copy of the initial positions
    initial_pos = positions.copy()
    reversed_chain = list(reversed(chain))
    root_pos = positions[chain[-1]].copy()
    for it in range(0, max_iter):
        # align target joint to target pos
        positions[chain[0]] = target_pos
        for chain_it_ind, chain_joint in enumerate(chain[1:]):
            child_ind = chain[chain_it_ind]
            direction = positions[child_ind]-positions[chain_joint]
            positions[chain_joint] = positions[child_ind] - arm_length[child_ind] * norm_vec(direction)
        # align root joint back to root pos
        positions[chain[-1]] = root_pos  
        for rev_chain_it_ind, chain_joint in enumerate(reversed_chain[1:]):
            parent_ind = reversed_chain[rev_chain_it_ind]
            direction = positions[parent_ind]-positions[chain_joint]
            positions[chain_joint] = positions[parent_ind] - arm_length[chain_joint] * norm_vec(direction)
        # break the loop if the condition is met
        if np.sum((positions[chain[0]]-target_pos) ** 2) < 1e-4:
            break

    # convert position to rotations, update the rotation_ref for each chain element
    parent_rotation = R.from_quat(wxyz_to_xyzw(rotation_ref[parents[reversed_chain[0]]]))
    parent_orientation_aft = R.from_quat([0, 0, 0, 1]) * parent_rotation
    parent_orientation_ori = parent_orientation_aft
    self_rotation_aft = R.from_quat([0, 0, 0, 1])
    new_rotations = []
    for ind, chain_joint in enumerate(reversed_chain[:-1]):
        chain_joint_child = reversed_chain[ind+1]
        ori_vec = initial_pos[chain_joint_child]-initial_pos[chain_joint]
        aft_vec = positions[chain_joint_child]-positions[chain_joint]
        q = from_to_rotation(ori_vec, aft_vec)
        self_rotation_ori = R.from_quat(wxyz_to_xyzw(rotation_ref[chain_joint]))
        # rotation is a local property
        self_rotation_aft = parent_orientation_aft.inv() * q * parent_orientation_ori * self_rotation_ori
        # keep record of the new rotations
        new_rotations.append(self_rotation_aft.as_quat().copy())
        # prepare for the next loop
        parent_orientation_aft = parent_orientation_aft * self_rotation_aft
        parent_orientation_ori = parent_orientation_ori * self_rotation_ori
    
    for ind, chain_joint in enumerate(reversed_chain[:-1]):
        rotation_ref[chain_joint] = xyzw_to_wxyz(new_rotations[ind])

def fix_root_barycenter(motion: Motion, left_foot=22, right_foot=18, avg_factor=3):
    positions = motion.get_joint_positions()
    nframes, njoints, nfeats = positions.shape
    avg_height = np.average(positions[:, [left_foot, right_foot], 1], axis=0)
    lcontact = positions[:, left_foot, 1].reshape((nframes))-np.repeat(avg_height[0]*avg_factor, nframes) < 0
    rcontact = positions[:, right_foot, 1].reshape((nframes))-np.repeat(avg_height[1]*avg_factor, nframes) < 0
    both_contact = lcontact.astype(np.int32) * rcontact.astype(np.int32)
    segment_start = 0
    for frame_ind in range(1, nframes):
        if both_contact[frame_ind-1] == 1 and both_contact[frame_ind] == 0 or frame_ind == nframes-1:
            # end of a segment
            fixed_barycenter_pos = np.average(positions[segment_start, [left_foot, right_foot]], axis=0)
            for mod_frame_ind in range(segment_start, frame_ind):
                barycenter_pos = np.average(positions[mod_frame_ind, [left_foot, right_foot]], axis=0)
                motion.positions[mod_frame_ind, 0] += fixed_barycenter_pos-barycenter_pos
        if both_contact[frame_ind-1] == 0 and both_contact[frame_ind] == 1:
            # start of a segment
            segment_start = frame_ind
    # modify the root positions
    # smooth the modified root motion
    root_barycenter_smooth_frames = 5
    motion.positions[:, 0] = gaussian_filter1d(motion.positions[:, 0], sigma=root_barycenter_smooth_frames, axis=0, mode='nearest')

def batch_quat_interpolation(_q1, _q2, alpha):
    q1 = _q1.copy()
    q2 = _q2.copy()
    batch_size, dim = q1.shape
    for b in range(batch_size):
        if np.dot(q1[b], q2[b]) < 0.0:
            q1[b] *= -1 # fix potential double cover problem
        q1[b] = q1[b] * alpha[b] + q2[b] * (1-alpha[b])
    return q1

def remove_footsliding(motion: Motion, lfoot=58, rfoot=54, avg_factor=2, contact_smooth_frames=3, minimum_contact_frames=5):
    '''
    Find out if the feet joint make contact with the ground.
    If so, lock the feet to the ground, utilizing inverse kinematics 
    to solve the relative positions of other joints.

    avg_factor: if the height of foot joint is greater than avg_height * avg_factor, it will be regarded as off-contact
    '''

    # modify the root positions
    fix_root_barycenter(motion, left_foot=lfoot, right_foot=rfoot, avg_factor=avg_factor)
    
    # Y stands for the height of the motion
    # XZ stands for the ground
    positions = motion.get_joint_positions()
    rotations = motion.rotations
    nframes, njoints, nfeats = positions.shape

    # locate the contact frames
    avg_height = np.average(positions[:, [lfoot, rfoot], 1], axis=0)
    # mark as off-contact once the height is greater than avg
    lcontact = positions[:, lfoot, 1].reshape((nframes))-np.repeat(avg_height[0] * avg_factor, nframes) < 0
    rcontact = positions[:, rfoot, 1].reshape((nframes))-np.repeat(avg_height[1] * avg_factor, nframes) < 0

    # precompute the armarture length for fabrik
    arm_length = []
    for joint_ind, joint_global_position in enumerate(positions[0]):
        if joint_ind == 0:
            arm_length.append(0)
        else:
            arm_length.append(np.sqrt(np.sum((joint_global_position-positions[0, motion.parents[joint_ind]])**2)))
    arm_length = np.array(arm_length)
    # precompute the kinematics chain for left and right feet
    lchain, rchain = [], []
    cur_joint = lfoot
    while motion.parents[cur_joint] != -1:
        lchain.append(cur_joint)
        cur_joint = motion.parents[cur_joint]
    cur_joint = rfoot
    while motion.parents[cur_joint] != -1:
        rchain.append(cur_joint)
        cur_joint = motion.parents[cur_joint]

    left_contact_slices = []
    right_contact_slices = []
    shift_flag_left, shift_flag_right = True, True
    last_shift_left, last_shift_right = 0, 0
    for frame_ind in range(1, nframes):
        if (lcontact[frame_ind-1] ^ lcontact[frame_ind]) or frame_ind == nframes-1:
            if shift_flag_left:
                shift_flag_left = False
                if frame_ind-last_shift_left >= minimum_contact_frames: # ignore contact that's too short
                    left_contact_slices.append((last_shift_left, frame_ind))
            else:
                shift_flag_left = True
                last_shift_left = frame_ind
        if (rcontact[frame_ind-1] ^ rcontact[frame_ind]) or frame_ind == nframes-1:
            if shift_flag_right:
                shift_flag_right = False
                if frame_ind-last_shift_left >= minimum_contact_frames:
                    right_contact_slices.append((last_shift_right, frame_ind))
            else:
                shift_flag_right = True
                last_shift_right = frame_ind

    # print(left_contact_slices)
    # print(right_contact_slices)

    left_lock_pos, right_lock_pos = [], []
    for slice_ind, left_slice in enumerate(left_contact_slices):
        start_frame, end_frame = left_slice
        lock_pos = positions[int((start_frame+end_frame)/2), lfoot]
        lock_pos[1] = 0.0
        left_lock_pos.append(lock_pos)
    for slice_ind, right_slice in enumerate(right_contact_slices):
        start_frame, end_frame = right_slice
        lock_pos = positions[int((start_frame+end_frame)/2), rfoot]
        lock_pos[1] = 0.0
        right_lock_pos.append(lock_pos)

    def fetch_lock_pos(fid, is_left):
        if is_left:
            for slice_ind, left_slice in enumerate(left_contact_slices):
                start_frame, end_frame = left_slice
                if start_frame <= fid and end_frame > fid:
                    return left_lock_pos[slice_ind]
            return positions[fid, lfoot]
        else:
            for slice_ind, right_slice in enumerate(right_contact_slices):
                start_frame, end_frame = right_slice
                if start_frame <= fid and end_frame > fid:
                    return right_lock_pos[slice_ind]
            return positions[fid, rfoot]

    left_foot_pos, right_foot_pos = np.zeros((nframes, 3)).astype(np.float32), np.zeros((nframes, 3)).astype(np.float32)
    for frame_ind in range(0, nframes):
        left_foot_pos[frame_ind] = fetch_lock_pos(frame_ind, True)
        right_foot_pos[frame_ind] = fetch_lock_pos(frame_ind, False)

    # use gaussian to smooth the expected lock position and free positions
    left_foot_pos = gaussian_filter1d(left_foot_pos, sigma=contact_smooth_frames, axis=0, mode='nearest')
    right_foot_pos = gaussian_filter1d(right_foot_pos, sigma=contact_smooth_frames, axis=0, mode='nearest')

    # use ik to reach for the smoothed positions
    for frame_ind in range(0, nframes):
        fabrik(arm_length, motion.parents, lchain, left_foot_pos[frame_ind],
               rotations[frame_ind], positions[frame_ind].copy())
        fabrik(arm_length, motion.parents, rchain, right_foot_pos[frame_ind],
               rotations[frame_ind], positions[frame_ind].copy())

    # # for ik test only
    # for frame_ind in range(0, nframes):
    #     fabrik(arm_length, motion.parents, lchain, np.zeros((3)),
    #            rotations[frame_ind], positions[frame_ind].copy())
    #     fabrik(arm_length, motion.parents, rchain, np.zeros((3)),
    #            rotations[frame_ind], positions[frame_ind].copy())

    return rotations

def concatenate_motion_clips(sequence_folder, smooth_frames = 3, lfoot = 62, rfoot = 57, avg_factor = 1.1):
    """
    Concatenate generated motion clips into a long sequence.
    
    Smooth out the root position, apply motion blending, remove foot sliding etc.
    """
    rotations_list = []
    root_pos_list = []

    for clip_idx, clip_path in enumerate(sorted(os.listdir(sequence_folder))):
        clip_path = os.path.join(sequence_folder, clip_path)
        motion = Motion.load_bvh(clip_path)
        rotations, root_pos = motion.rotations, motion.positions[:, 0]
        rotations_list.append(rotations)
        root_pos_list.append(root_pos)
    T_pose_template = motion.copy()

    # Concatenate all clips
    ori_root_pos = np.concatenate(root_pos_list, axis=0)
    smooth_root_pos = gaussian_filter1d(ori_root_pos, smooth_frames, axis=0, mode='nearest')

    # Blend the positions
    for clip_idx in range(1, len(rotations_list)):
        prev_rotations = rotations_list[clip_idx - 1]
        next_rotations = rotations_list[clip_idx]
        dead_blending(prev_rotations, next_rotations)

    smooth_rotations = np.concatenate(rotations_list, axis=0)

    T_pose_template.rotations = smooth_rotations
    T_pose_template.positions = np.zeros((smooth_rotations.shape[0], smooth_rotations.shape[1], 3))
    T_pose_template.positions[:, 0] = smooth_root_pos

    # Remove foot sliding
    T_pose_template.rotations = remove_footsliding(T_pose_template, lfoot, rfoot, avg_factor)

    T_pose_template.export(f'results/{os.path.basename(sequence_folder)}_seq.bvh')

def xyzw_to_wxyz(quat):
    q = quat.copy()
    w = q[3]
    q[3] = q[2]
    q[2] = q[1]
    q[1] = q[0]
    q[0] = w
    return q

def wxyz_to_xyzw(quat):
    q = quat.copy()
    w = q[0]
    q[0] = q[1]
    q[1] = q[2]
    q[2] = q[3]
    q[3] = w
    return q
