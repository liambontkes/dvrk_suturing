# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import jupyros as jr
import rospy
import numpy as np
from sensor_msgs import msg
import cv2
import cv_bridge
from copy import deepcopy
import ipywidgets as widgets
import PIL.Image
from cStringIO import StringIO
import matplotlib.pyplot as plt
import dvrk
import PyKDL
import tf
import time
from tf_conversions import posemath
import utils

rospy.init_node('notebook')
rospy.get_published_topics()

# +
bridge = cv_bridge.CvBridge()
left_image = None
left_image_msg = None
left_camera_info = None

right_image = None
right_image_msg = None
right_camera_info = None

def left_image_callback(im_msg):
    global left_image, left_image_msg
    left_image = bridge.imgmsg_to_cv2(im_msg, desired_encoding='rgb8')
    left_image_msg = im_msg
    
def right_image_callback(im_msg):
    global right_image, right_image_msg
    right_image = bridge.imgmsg_to_cv2(im_msg, desired_encoding='rgb8')
    right_image_msg = im_msg
    
def left_camera_info_callback(camera_info_msg):
    global left_camera_info
    left_camera_info = camera_info_msg
    
def right_camera_info_callback(camera_info_msg):
    global right_camera_info
    right_camera_info = camera_info_msg
    
jr.subscribe('/stereo/left/image_raw', msg.Image, left_image_callback)
jr.subscribe('/stereo/left/camera_info', msg.CameraInfo, left_camera_info_callback)
jr.subscribe('/stereo/right/image_raw', msg.Image, right_image_callback)
jr.subscribe('/stereo/right/camera_info', msg.CameraInfo, right_camera_info_callback)

while left_image is None or right_image is None:
    time.sleep(0.5)
# -

plt.imshow(np.hstack((left_image, right_image)))

tf_listener = tf.TransformListener()
time.sleep(1)
tf_listener.getFrameStrings()

psm1 = dvrk.psm('PSM1')
ecm = dvrk.ecm('ECM')
while ecm.get_current_position() == PyKDL.Frame() or ecm.get_desired_position() == PyKDL.Frame():
    time.sleep(0.5)

ECM_STARTING_JOINT_POS = np.asarray([-0.15669435,  0.17855662,  0.07069676,  0.17411496])
ecm.move_joint(ECM_STARTING_JOINT_POS)

PSM_HOME_POS = np.asarray([0., 0., 0.05, 0., 0., 0.])
psm1.move_joint(PSM_HOME_POS)

# +
import image_geometry
utils = None
import utils
stereo_model = image_geometry.StereoCameraModel()
stereo_model.fromCameraInfo(left_camera_info, right_camera_info)

tf_cam_to_world = utils.tf_to_pykdl_frame(tf_listener.lookupTransform('simworld', 'simcamera', rospy.Time()))
tf_world_to_psm1 = \
    utils.PSM_J1_TO_BASE_LINK_TF * utils.tf_to_pykdl_frame(tf_listener.lookupTransform('J1_PSM1', 'simworld', rospy.Time()))

objects, frame = utils.get_points_and_img(left_image_msg, right_image_msg, stereo_model, tf_cam_to_world)
plt.figure(figsize=(12, 5))
plt.imshow(frame)
# -

objects

# pair up points that are across from each other
# x is *more or less* the axis along the wound
paired_pts = []
for pt in objects:
    objects.remove(pt)
    pt2 = min(objects, key=lambda obj : abs(obj.x() - pt.x()))
    objects.remove(pt2)
    paired_pts.append(
        (max(pt, pt2, key=lambda p: p.y()), min(pt, pt2, key=lambda p: p.y())))
paired_pts

PSM_HOME_POS = np.asarray([0., 0., 0.05, 0., 0., 0.])
psm1.move_joint(PSM_HOME_POS)
psm1.close_jaw()

# +
# calculate a desired pose
# this is really pushing my first year linear algebra skills
# we want the insertion to go from the min y point to the max y point
import math

NEEDLE_Z_OFFSET = -0.008
NEEDLE_Y_OFFSET = 0.0010
CIRCLE_Z_OFFSET = 0.002
NEEDLE_RADIUS = 0.0115


# the desired rotation is calculated by setting the entry-to-exit vector of the suture as the x-axis vector,
# setting x-axis cross (0, 0, 1) as the z-axis vector, and setting the y-axis vector to the x-axis cross z-axis.
def calculate_desired_entry_pose(entry_and_exit_point):
    entry_to_exit_vector = entry_and_exit_point[1] - entry_and_exit_point[0]
    entry_to_exit_vector.Normalize()
    desired_z_vector = - entry_to_exit_vector * PyKDL.Vector(0, 0, 1)
    desired_y_vector = entry_to_exit_vector
    desired_x_vector = - desired_z_vector * desired_y_vector
    
    desired_rotation = \
        PyKDL.Rotation(desired_x_vector, desired_y_vector, desired_z_vector)
#     desired_rotation.DoRotZ(0.2)
    
    desired_position = entry_and_exit_point[0] + (desired_z_vector * NEEDLE_Z_OFFSET)
    return PyKDL.Frame(desired_rotation, desired_position)
   
desired_pose = calculate_desired_entry_pose(paired_pts[0])
psm1.move(tf_world_to_psm1 * desired_pose)


# +
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

def calculate_circular_pose(entry_and_exit_points, entry_pose, circular_progress_radians, circle_radius=NEEDLE_RADIUS):
    # this sets the desired rotation and translation to a pose around the circle with diameter 
    # consisting of entry_and_exit_points and rotation CW about the z-axis of entry_pose such that the
    # x-axis is tangent to the circle
    new_orientation = deepcopy(entry_pose.M)
    new_orientation.DoRotZ(circular_progress_radians)
    
    circle_center = fit_circle_to_points_and_radius(entry_pose, entry_and_exit_points, circle_radius)
    print("circle_center={}, circle_radius={}".format(circle_center, circle_radius))
    desired_angle_radial_vector = new_orientation * PyKDL.Vector(0, - circle_radius, 0)
    new_position = desired_angle_radial_vector + circle_center \
                   + (new_orientation.UnitY() * NEEDLE_Y_OFFSET)
    
    return PyKDL.Frame(new_orientation, new_position)

# for rads in np.arange(0, 3.4, 0.2):
#     insertion_pose = calculate_circular_pose(paired_pts[0], desired_pose, rads)
#     psm1.move(tf_world_to_psm1 * insertion_pose)
    
from utils import CircularMotion
reload(utils)

cm = CircularMotion(psm1, tf_world_to_psm1, NEEDLE_RADIUS, paired_pts[0], desired_pose, 0, 3.4)

while not cm.is_done():
    cm.step()


# -

psm1.open_jaw()
prepare_pickup_circle_pose = PyKDL.Frame(desired_pose.M, desired_pose.p 
                                         + desired_pose.M.Inverse() * PyKDL.Vector(0, 0.01, 0))
terminal_rads = 3.4
pickup_rads = terminal_rads + np.pi - 0.25
# rotate further than necessary in order to force the wrist to flip over
# TODO: this movement causes erratic motion
opposite_pose = calculate_circular_pose(paired_pts[0], prepare_pickup_circle_pose, 
                                        terminal_rads + np.pi + 0.25, NEEDLE_RADIUS + 0.005)
psm1.move(tf_world_to_psm1 * opposite_pose)

# +

# move to the actual pickup position
opposite_pose = calculate_circular_pose(paired_pts[0], desired_pose, pickup_rads)
psm1.move(tf_world_to_psm1 * opposite_pose)
psm1.close_jaw()

# +
# use calculate_circular_pose to do the extraction
for rads in np.arange(pickup_rads, pickup_rads + 3.0, 0.2):
    opposite_pose = calculate_circular_pose(paired_pts[0], desired_pose, rads)
    psm1.move(tf_world_to_psm1 * opposite_pose)
    
terminal_rads = rads
# -

psm1.open_jaw()
pickup_pose = calculate_circular_pose(paired_pts[1], desired_pose, terminal_rads - np.pi + 0.2)
psm1.dmove(PyKDL.Vector(0, 0, 0.02))
psm1.move(tf_world_to_psm1 * pickup_pose)
psm1.close_jaw()
psm1.move_joint(PSM_HOME_POS)

# +
# setup for insertion
desired_pose = calculate_desired_entry_pose(paired_pts[1])
psm1.move(tf_world_to_psm1 * desired_pose)


# insertion
for rads in np.arange(0, 3.4, 0.2):
    insertion_pose = calculate_circular_pose(paired_pts[1], desired_pose, rads)
    psm1.move(tf_world_to_psm1 * insertion_pose)
    
print("terminal position in rads: {}".format(rads))

# setup and grasp needle for extraction
psm1.open_jaw()
psm1.dmove(PyKDL.Vector(0, 0, 0.02))
terminal_rads = rads
pickup_rads = terminal_rads + np.pi - 0.25
# rotate further than necessary in order to force the wrist to flip over
opposite_pose = calculate_circular_pose(paired_pts[1], desired_pose, terminal_rads + np.pi + 0.25)
psm1.move(tf_world_to_psm1 * opposite_pose)
# move to the actual pickup position
opposite_pose = calculate_circular_pose(paired_pts[1], desired_pose, pickup_rads)
psm1.move(tf_world_to_psm1 * opposite_pose)
psm1.close_jaw()

# extraction
for rads in np.arange(pickup_rads, pickup_rads + 3.0, 0.2):
    opposite_pose = calculate_circular_pose(paired_pts[1], desired_pose, rads)
    psm1.move(tf_world_to_psm1 * opposite_pose)
    
terminal_rads = rads

# grasp base
psm1.open_jaw()
pickup_pose = calculate_circular_pose(paired_pts[1], desired_pose, terminal_rads - np.pi)
psm1.dmove(PyKDL.Vector(0, 0, 0.02))
psm1.move(tf_world_to_psm1 * pickup_pose)
psm1.close_jaw()
psm1.move_joint(PSM_HOME_POS)
# -


