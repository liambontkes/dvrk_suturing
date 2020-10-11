#!/usr/bin/env python
import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from auto_cam.msg import FeaturePoints
from opencv_apps.msg import Point2D
from enum import Enum
import cv2
import numpy as np
import imutils
import sys
import csv
import math
from skimage import transform as sktf

class FeatureType(Enum):
    BALL = 0
    BOWL = 1


def contour_centre(contour):
    M = cv2.moments(contour)
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])
    return (x, y)

class ImageFeature:
    def __init__(self, pos, color, contour):
        self.pos = pos
        self.contour = contour
        self.color = color
     
    def __str__(self):
        return "ImageFeature pos={}, color={}, area={}".format(
                self.pos, self.color, cv2.contourArea(self.contour))

    def __repr__(self):
        return self.__str__()


class feature_processor:

    def __init__(self, feature_files, log_verbose=False):
        self.log_verbose = False
        # Set hsv lower and upper limits
        self.StoreHSVRanges(feature_files)
        # Declare adjustment values to match cv2 hsv value storage
        self.hsv_adjustment = np.array(
            [1.0 / 2.0, 255.0 / 100.0, 255.0 / 100.0])

        self.min_contour_area = 10


        # Declare a cv to ros bridge
        self.bridge = CvBridge()

    def StoreHSVRanges(self, feature_files):
        hsv_ranges = np.empty((1,  2, 3))
        for feature in feature_files:
            feature_range = np.genfromtxt(feature, delimiter=',')
            range_reshaped = np.reshape(feature_range,  (1, 2, 3))
            hsv_ranges = np.append(hsv_ranges, range_reshaped, axis=0)

        # Delete empty row
        hsv_ranges = np.delete(hsv_ranges,  0, axis=0)

        self.hsv_ranges = hsv_ranges
        self.n_features = hsv_ranges.shape[0]


    def GetShapeSize(self, points):
        # Store area
        if (points.shape[0] == 0):
            rospy.logerr("No feature points to find size of!")
            size = 0
        if(points.shape[0] == 1):
            # Size is zero
            size = 0
        if (points.shape[0] == 2):
            # Store length line instead of area
            size = np.linalg.norm(
                points[0, :] - points[1, :])
        else:
            size = cv2.contourArea(np.float32(points))
        rospy.loginfo("Size: "+ str(size))
        return size

    def PrepareImage(self, ros_image):
        # try catch block to capture exception handling
        try:
            # Convert ROS message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(
                ros_image, desired_encoding='rgb8')
        except CvBridgeError as e:
            rospy.loginfo(e)

        # Draw a circle outline at the centre of the frame
        height, width = frame.shape[0:2]
        if self.log_verbose:
            rospy.loginfo("Height: " + str(height) + " Width: " + str(width))
        cv2.circle(frame, (int(width / 2), int(height / 2)),
                   radius=2, color=(0, 0, 0), thickness=1)

        # Covert OpenCV image into gray scale
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        return frame, hsv


    def FindContours(self, hsv, lower_range, upper_range):
        # Masks the input frame using the HSV upper and lower range
        mask = cv2.inRange(hsv, lower_range * self.hsv_adjustment,
                           upper_range * self.hsv_adjustment)

        # Create contours of the segmented diagram from the HSV
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        return contours


    def FindImageFeatures(self, ros_image):
        # Prepare image for processing
        frame, hsv = self.PrepareImage(ros_image)

        features = []
        for f in range(self.n_features):
            # The lower and upper ranges of HSV recognition for this feature
            lower_range = self.hsv_ranges[f, 0, :]
            upper_range = self.hsv_ranges[f, 1, :]

            contours = self.FindContours(hsv, lower_range, upper_range)

            # Iterate through all contours
            for c in contours:

                # Skip contours with areas that are too small
                if (cv2.contourArea(c) < self.min_contour_area):
                    continue

                # Draw the contour lines
                cv2.drawContours(frame, [c], -1, (0, 0, 0), 1)

                # # Find the x and y coordinates of the centroid of the object.
                # center = contour_centre(c)
                center, _ = cv2.minEnclosingCircle(c)

                cx = int(center[0])
                cy = int(center[1])

                features.append(ImageFeature(pos=(cx, cy), color=f, contour=c))

                # Creates a circle at the centroid point
                cv2.circle(frame, (cx, cy), 3, (0, 0, 0), -1)
#                print("features pos:", features.pos[0],features.pos[2])
        
        sorted(features,key=lambda p: p.pos[0])

        return features, frame