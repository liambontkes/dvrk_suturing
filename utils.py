import rospy
import numpy as np
from sensor_msgs import msg
import cv2
import cv_bridge
from copy import deepcopy
import dvrk
import PyKDL
import tf
import image_geometry
from feature_processor import feature_processor, FeatureType
from tf_conversions import posemath
from math import pi
import dvrk
from collections import OrderedDict 
import math 

'''
Some useful methods and constants for picking up a ball with dVRK and CoppeliaSim
'''

PSM_J1_TO_BASE_LINK_ROT = PyKDL.Rotation.RPY(pi / 2, - pi, 0)
PSM_J1_TO_BASE_LINK_TF = PyKDL.Frame(PSM_J1_TO_BASE_LINK_ROT, PyKDL.Vector())

BLUE_CIRCLE_FEAT_PATH = './blue_circle.csv'

FEAT_PATHS = [BLUE_CIRCLE_FEAT_PATH]

# The empirically determined needle Z- and Y- offsets from the gripper frame
NEEDLE_Z_OFFSET = -0.008
NEEDLE_Y_OFFSET = 0.0010
# The Z offset of the plane on which the suture throw circle lies
CIRCLE_Z_OFFSET = 0.002

# the empirically determined radius of the needle (the needle diameter 
# is slightly less than the bounding box width thanks to its shape)
NEEDLE_RADIUS = 0.0115


# TODO: now that the camera is right side up, maybe this can be changed
CV_TO_CAM_FRAME_ROT = np.asarray([
    [-1, 0, 0], 
    [0, -1, 0],
    [0, 0, 1]
])


def clamp_image_coords(pt, im_shape):
    return tuple(np.clip(pt, (0, 0), np.array(im_shape)[:2] - np.array([1, 1])))


def get_points_and_img(left_image_msg, right_image_msg, stereo_cam_model, cam_to_world_tf):
    # this gets the position of the red ball thing in the camera frame
    # and the image with X's on the desired features
    fp = feature_processor(FEAT_PATHS)
    left_feats, left_frame = fp.FindImageFeatures(left_image_msg)
    right_feats, right_frame = fp.FindImageFeatures(right_image_msg)

    matched_feats = []

    for left_feat in left_feats:
        matched_feats.append((left_feat, 
                              min(right_feats, 
                                  key=lambda right_feat: (right_feat.pos[0] - left_feat.pos[0]) ** 2 \
                                                       + (right_feat.pos[1] - left_feat.pos[1]) ** 2)))

    objects = []
    for left_feat, right_feat in matched_feats:
        disparity = abs(left_feat.pos[0] - right_feat.pos[0])
        pos_cv = stereo_cam_model.projectPixelTo3d(left_feat.pos, float(disparity))
        pos_cam = np.matmul(CV_TO_CAM_FRAME_ROT, pos_cv)
        print(pos_cam)
        pos = PyKDL.Vector(*pos_cam)
        print(pos)
        objects.append(cam_to_world_tf * pos)
    return objects, np.hstack((left_frame, right_frame))


def tf_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)

def fit_circle_to_points_and_radius(circle_plane_pose, points, radius):
    # taken from http://mathforum.org/library/drmath/view/53027.html
    # because the circle plane pose has the z-axis perpendicular to the desired circle,
    # we can zero out the z-axis component of our dots 
    p1 = circle_plane_pose.Inverse() * points[0]
    p1 = PyKDL.Vector(p1.x(), p1.y(), 0)
    p2 = circle_plane_pose.Inverse() * points[1]
    p2 = PyKDL.Vector(p2.x(), p2.y(), 0)
    
    q = (p2 - p1).Norm()
    mean_x = np.mean([p1.x(), p2.x()])
    mean_y = np.mean([p1.y(), p2.y()])
    
    # EXTREMELY BRITTLE, dependent on order that `points` are in
    x = mean_x - (math.sqrt(radius ** 2 - (q / 2) ** 2) * (p1.y() - p2.y())) / q
    y = mean_y - (math.sqrt(radius ** 2 - (q / 2) ** 2) * (p2.x() - p1.x())) / q
    
    return circle_plane_pose * PyKDL.Vector(x, y, 0)


def calculate_circular_pose(entry_and_exit_points, entry_pose, circular_progress_radians):
    # this sets the desired rotation and translation to a pose around the circle with diameter 
    # consisting of entry_and_exit_points and rotation CW about the z-axis of entry_pose such that the
    # x-axis is tangent to the circle
    new_orientation = deepcopy(entry_pose.M)
    new_orientation.DoRotZ(circular_progress_radians)
    
    circle_center = fit_circle_to_points_and_radius(entry_pose, entry_and_exit_points, NEEDLE_RADIUS)
    circle_radius = NEEDLE_RADIUS
    print("circle_center={}, circle_radius={}".format(circle_center, circle_radius))
    desired_angle_radial_vector = new_orientation * PyKDL.Vector(0, - circle_radius, 0)
    new_position = desired_angle_radial_vector + circle_center \
                   + (new_orientation.UnitY() * NEEDLE_Y_OFFSET)
    
    return PyKDL.Frame(new_orientation, new_position)


def vector_eps_eq(lhs, rhs):
    return bool((lhs - rhs).Norm() < 0.001)


def set_arm_dest(arm, dest_pose):
    if arm.get_desired_position() != dest_pose:
        arm.move(dest_pose, blocking=False)


class CircularMotion:
    def __init__(self, psm, world_to_psm_tf, circle_radius, points, circle_pose, 
                 start_rads, end_rads, step_rads=0.2):
        self.psm = psm
        self.world_to_psm_tf = world_to_psm_tf
        self.poses = []
        for rads in np.arange(start_rads, end_rads, step_rads):
            self.poses.append(self.world_to_psm_tf * calculate_circular_pose(points, circle_pose, rads))
        self.pose_idx = 0
        self.done = False

    def step(self):
        if self.done:
            return

        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self.poses[self.pose_idx].p):
            if self.pose_idx < len(self.poses) - 1:
                self.pose_idx += 1
            else:
                self.done = True

        set_arm_dest(self.psm, self.poses[self.pose_idx])
        rospy.loginfo("Moving to pose {} out of {}".format(self.pose_idx, len(self.poses)))

    
    def is_done(self):
       return self.done 


