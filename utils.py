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

'''
Some useful methods and constants for picking up a ball with dVRK and CoppeliaSim
'''

PSM_J1_TO_BASE_LINK_ROT = PyKDL.Rotation.RPY(pi / 2, - pi, 0)
PSM_J1_TO_BASE_LINK_TF = PyKDL.Frame(PSM_J1_TO_BASE_LINK_ROT, PyKDL.Vector())

BLUE_CIRCLE_FEAT_PATH = './blue_circle.csv'

FEAT_PATHS = [BLUE_CIRCLE_FEAT_PATH]

# TODO: now that the camera is right side up, maybe this can be changed
CV_TO_CAM_FRAME_ROT = np.asarray([
    [-1, 0, 0], 
    [0, -1, 0],
    [0, 0, 1]
])

class Object3d:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color
    
    def __str__(self):
        return "<Object3d pos: {} color: {}>".format(self.pos, self.color)

    def __repr__(self):
        return self.__str__()


def clamp_image_coords(pt, im_shape):
    return tuple(np.clip(pt, (0, 0), np.array(im_shape)[:2] - np.array([1, 1])))


def get_objects_and_img(left_image_msg, right_image_msg, stereo_cam_model, cam_to_world_tf):
    # this gets the position of the red ball thing in the camera frame
    # and the image with X's on the desired features
    fp = feature_processor(FEAT_PATHS)
    left_feats, left_frame = fp.FindImageFeatures(left_image_msg)
    right_feats, right_frame = fp.FindImageFeatures(right_image_msg)

    matched_feats = []

    for left_feat in left_feats:
        same_color_feats = filter(lambda right_feat: right_feat.color == left_feat.color, 
                                  right_feats)

        if not same_color_feats:
            rospy.logwarn(
                "Failed to match left detection {} to a right detection!".format(left_feat))
            continue

        matched_feats.append((left_feat, 
                              min(same_color_feats, 
                                  key=lambda right_feat: (right_feat.pos[0] - left_feat.pos[0]) ** 2 \
                                                       + (right_feat.pos[1] - left_feat.pos[1]) ** 2)))

    objects = []
    for left_feat, right_feat in matched_feats:
        disparity = abs(left_feat.pos[0] - right_feat.pos[0])
        pos_cv = stereo_cam_model.projectPixelTo3d(left_feat.pos, float(disparity))
        # there's a fixed rotation to convert this to the camera coordinate frame
        pos_cam = np.matmul(CV_TO_CAM_FRAME_ROT, pos_cv)
        pos = None
        if cam_to_world_tf is not None:
            pos = cam_to_world_tf * PyKDL.Vector(*pos_cam)
        else:
            pos = PyKDL.Vector(*pos_cam)

        if left_feat.color != right_feat.color:
            rospy.loginfo("Color mismatch between left and right detection")

        objects.append(Object3d(pos, left_feat.color))
    return objects, np.hstack((left_frame, right_frame))


def tf_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)


class World:
    def __init__(self, all_detections):
        self.objects = []

    def __str__(self):
        return "<World objects: {}\nbowl: {}>".format(self.objects, self.bowl)

    def __repr__(self):
        return self.__str__()