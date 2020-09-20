from utils import *
from enum import Enum

class SuturingState(Enum):
    HOME = 0
    PREPARE_INSERTION = 1
    INSERTION = 2
    OVERROTATE = 3
    PREPARE_EXTRACTION = 4
    EXTRACTION = 5
    PICKUP = 6
    DONE = 7


class SuturingStateMachine:

    def _home_state(self):
        if self.psm.desired_joint_position() != PSM_HOME_JOINT_POS:
            self.psm.move(PSM_HOME_JOINT_POS, blocking=False)


    def _home_next(self):
        if self.psm._arm__goal_reached and \
            np.max(self.psm.current_joint_position() - PSM_HOME_JOINT_POS) < 0.005:
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
        if arm_pos_reached(self.psm, self.circle_pose.p):
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
        if self.circular_motion.is_done():
            self.circular_motion = None
            return SuturingState.MOVE_UP
        else:
            return SuturingState.INSERTION


    def _overrotate_state(self):
        # an upward motion and a 'over-rotation' to get the gripper in the right pose for 
        # extracting the needle
        overrotation_circle_pose = PyKDL.Frame(self.circle_pose.M, self.circle_pose.p 
                                             + self.circle_pose.M.Inverse() * PyKDL.Vector(0, 0.015, 0))
        self.overrotation_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx],
                                                         self.insertion_rads + np.pi + 0.25, 
                                                         NEEDLE_RADIUS + 0.005)
        set_arm_dest(self.psm, self.overrotation_pose)


    def _overrotate_next(self):
        if arm_pos_reached(self.psm, self.overrotation_pose.p):
            self.overrotation_pose = None
            return SuturingState.PREPARE_EXTRACTION
        else:
            return SuturingState.OVERROTATE

    
    def _prepare_extraction_state(self):
        pickup_rads = self.insertion_rads + np.pi - 0.25
        opposite_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx], 
                                                self.circle_pose, pickup_rads)
        self.prepare_extraction_pose = opposite_pose
        set_arm_dest(self.psm, self.prepare_extraction_pose)
            

    def _prepare_extraction_next(self):
        if arm_pos_reached(self.psm, self.prepare_extraction_pose.p):
            self.prepare_extraction_pose = None
            return SuturingState.EXTRACTION
        else:
            return SuturingState.PREPARE_EXTRACTION


    def _extraction_state(self):
        if self.circular_motion is None:
            self.circular_motion = CircularMotion(self.psm, self.tf_world_to_psm, NEEDLE_RADIUS,
                                                  self.paired_pts[self.paired_pts_idx],
                                                  self.circular_motion, 
                                                  self.insertion_rads + np.pi - 0.25,
                                                  # TODO: tweak this value
                                                  self.insertion_rads + 2 * np.pi - 0.25)
        self.circular_motion.step()


    def _extraction_next(self):
        if self.circular_motion.is_done():
            self.circle_pose = None
            self.circular_motion = None
            self.paired_pts_idx += 1
            return SuturingState.HOME
        else:
            return SuturingState.EXTRACTION


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

        self.state_funs = {
            SuturingState.HOME : self._home_state,
            SuturingState.PREPARE_INSERTION : self._prepare_insertion_state,
            SuturingState.INSERTION : self._insertion_state,
            SuturingState.OVERROTATE : self._overrotate_state,
            SuturingState.PREPARE_EXTRACTION : self._prepare_extraction_state,
            SuturingState.EXTRACTION : self._extraction_state
        }

        self. next_funs = {
            SuturingState.HOME : self._home_next,
            SuturingState.PREPARE_INSERTION : self._prepare_insertion_next,
            SuturingState.INSERTION : self._insertion_next,
            SuturingState.OVERROTATE : self._overrotate_next,
            SuturingState.PREPARE_EXTRACTION : self._prepare_extraction_next,
            SuturingState.EXTRACTION : self._extraction_next
        }
        