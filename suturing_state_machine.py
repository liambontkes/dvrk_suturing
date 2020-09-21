from utils import *
from enum import Enum

class SuturingState(Enum):
    HOME = 0
    PREPARE_INSERTION = 1
    INSERTION = 2
    RELEASE_NEEDLE = 3
    OVERROTATE = 4
    PREPARE_EXTRACTION = 5
    GRASP_NEEDLE = 6
    EXTRACTION = 7
    RELEASE_NEEDLE_2 = 8
    PICKUP = 9
    GRASP_NEEDLE_2 = 10
    DONE = 11


class SuturingStateMachine:

    def jaw_fully_open(self):
        return True if self.psm.get_current_jaw_position() >= math.pi / 3 else False 

    def jaw_fully_closed(self):
        return True if self.psm.get_current_jaw_position() <= 0 else False

    def _home_state(self):
        if (self.psm.get_desired_joint_position() != PSM_HOME_JOINT_POS).any():
            self.psm.move_joint(PSM_HOME_JOINT_POS, blocking=False)


    def _home_next(self):
        if self.psm._arm__goal_reached and \
            np.max(self.psm.get_current_joint_position() - PSM_HOME_JOINT_POS) < 0.005:
            if self.paired_pts_idx < len(self.paired_pts):
                return SuturingState.PREPARE_INSERTION
            else:
                return SuturingState.DONE
        else:
            return SuturingState.HOME


    def _prepare_insertion_state(self):
        if self.circle_pose is None:
            self.circle_pose = calculate_desired_entry_pose(self.paired_pts[self.paired_pts_idx])
        set_arm_dest(self.psm, self.tf_world_to_psm * self.circle_pose)


    def _prepare_insertion_next(self):
        if arm_pos_reached(self.psm, self.tf_world_to_psm * self.circle_pose.p):
            return SuturingState.INSERTION
        else:
            return SuturingState.PREPARE_INSERTION

    
    def _insertion_state(self):
        if self.circular_motion is None:
            self.circular_motion = CircularMotion(self.psm, self.tf_world_to_psm, NEEDLE_RADIUS,
                                                  self.paired_pts[self.paired_pts_idx],
                                                  self.circle_pose, 0, self.insertion_rads)
        self.circular_motion.step()

    
    def _insertion_next(self):
        if self.circular_motion is not None and self.circular_motion.is_done():
            self.circular_motion = None
            return SuturingState.RELEASE_NEEDLE
        else:
            return SuturingState.INSERTION


    def _release_needle_state(self):
        if self.psm.get_desired_jaw_position() <= 0.:
            self.psm.open_jaw(blocking=False)

    
    def _release_needle_next(self):
        if self.jaw_fully_open():
            return SuturingState.OVERROTATE
        else:
            return SuturingState.RELEASE_NEEDLE


    def _overrotate_state(self):
        # an upward motion and a 'over-rotation' to get the gripper in the right pose for 
        # extracting the needle
        overrotation_circle_pose = PyKDL.Frame(self.circle_pose.M, self.circle_pose.p 
                                             + self.circle_pose.M.Inverse() * PyKDL.Vector(0, 0.015, 0))
        self.overrotation_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx],
                                                         overrotation_circle_pose,
                                                         self.insertion_rads + np.pi + 0.25, 
                                                         NEEDLE_RADIUS + 0.005)
        set_arm_dest(self.psm, self.tf_world_to_psm * self.overrotation_pose)


    def _overrotate_next(self):
        if arm_pos_reached(self.psm, self.tf_world_to_psm * self.overrotation_pose.p):
            self.overrotation_pose = None
            return SuturingState.PREPARE_EXTRACTION
        else:
            return SuturingState.OVERROTATE

    
    def _prepare_extraction_state(self):
        pickup_rads = self.insertion_rads + np.pi - 0.25
        opposite_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx], 
                                                self.circle_pose, pickup_rads)
        self.prepare_extraction_pose = opposite_pose
        set_arm_dest(self.psm, self.tf_world_to_psm * self.prepare_extraction_pose)
            

    def _prepare_extraction_next(self):
        if arm_pos_reached(self.psm, self.tf_world_to_psm * self.prepare_extraction_pose.p):
            self.prepare_extraction_pose = None
            return SuturingState.GRASP_NEEDLE
        else:
            return SuturingState.PREPARE_EXTRACTION


    def _grasp_needle_state(self):
        if self.psm.get_desired_jaw_position() >= 0.:
            self.psm.close_jaw(blocking=False)

    
    def _grasp_needle_next(self):
        if self.jaw_fully_closed():
            return SuturingState.EXTRACTION
        else:
            return SuturingState.GRASP_NEEDLE


    def _extraction_state(self):
        if self.circular_motion is None:
            self.circular_motion = CircularMotion(self.psm, self.tf_world_to_psm, NEEDLE_RADIUS,
                                                  self.paired_pts[self.paired_pts_idx],
                                                  self.circle_pose, 
                                                  self.insertion_rads + np.pi - 0.25,
                                                  # TODO: tweak this value
                                                  self.insertion_rads + 2 * np.pi - 0.25)
        self.circular_motion.step()


    def _extraction_next(self):
        if self.circular_motion is not None and self.circular_motion.is_done():
            self.circle_motion = None
            return SuturingState.RELEASE_NEEDLE_2
        else:
            return SuturingState.EXTRACTION

    
    def _release_needle_2_next(self):
        if self.jaw_fully_open():
            return SuturingState.PICKUP
        else:
            return SuturingState.RELEASE_NEEDLE_2


    def _pickup_state(self):
        if self.pickup_pose is None:
            self.pickup_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx], 
                                                       self.circle_pose,
                                                       self.extraction_rads - np.pi + 0.2)
        set_arm_dest(self.psm, self.tf_world_to_psm * self.pickup_pose)

    def _pickup_next(self):
        if arm_pos_reached(self.psm, self.tf_world_to_psm * self.pickup_pose.p):
            self.pickup_pose = None
            return SuturingState.GRASP_NEEDLE_2
        else:
            return SuturingState.PICKUP


    def _grasp_needle_2_next(self):
        if self.jaw_fully_closed():
            self.circle_pose = None
            self.circular_motion = None
            self.paired_pts_idx += 1
            return SuturingState.HOME
        else:
            return SuturingState.GRASP_NEEDLE_2


    def is_done(self):
        return self.state == SuturingState.DONE


    def run_once(self):
        self.state = self.next_funs[self.state]()

        if self.state == SuturingState.DONE:
            return

        rospy.loginfo("Executing state {}".format(self.state))
        self.state_funs[self.state]()


    def __init__(self, psm, tf_world_to_psm, paired_pts, insertion_rads=3.4, extraction_rads=3.0):
        self.psm = psm
        self.tf_world_to_psm = tf_world_to_psm
        self.paired_pts = paired_pts
        self.paired_pts_idx = 0
        self.insertion_rads = insertion_rads
        self.extraction_rads = extraction_rads
        self.state = SuturingState.HOME

        # evidence that i did not spend any time on good code design
        self.circle_pose = None
        self.circular_motion = None
        self.overrotation_pose = None
        self.prepare_extraction_pose = None
        self.pickup_pose = None

        self.state_funs = {
            SuturingState.HOME : self._home_state,
            SuturingState.PREPARE_INSERTION : self._prepare_insertion_state,
            SuturingState.INSERTION : self._insertion_state,
            SuturingState.RELEASE_NEEDLE : self._release_needle_state,
            SuturingState.OVERROTATE : self._overrotate_state,
            SuturingState.PREPARE_EXTRACTION : self._prepare_extraction_state,
            SuturingState.GRASP_NEEDLE : self._grasp_needle_state,
            SuturingState.EXTRACTION : self._extraction_state,
            SuturingState.RELEASE_NEEDLE_2: self._release_needle_state,
            SuturingState.PICKUP : self._pickup_state,
            SuturingState.GRASP_NEEDLE_2 : self._grasp_needle_state
        }

        self. next_funs = {
            SuturingState.HOME : self._home_next,
            SuturingState.PREPARE_INSERTION : self._prepare_insertion_next,
            SuturingState.INSERTION : self._insertion_next,
            SuturingState.RELEASE_NEEDLE : self._release_needle_next,
            SuturingState.OVERROTATE : self._overrotate_next,
            SuturingState.PREPARE_EXTRACTION : self._prepare_extraction_next,
            SuturingState.GRASP_NEEDLE : self._grasp_needle_next,
            SuturingState.EXTRACTION : self._extraction_next,
            SuturingState.RELEASE_NEEDLE_2 : self._release_needle_2_next,
            SuturingState.PICKUP : self._pickup_next,
            SuturingState.GRASP_NEEDLE_2 : self._grasp_needle_2_next
        }
        