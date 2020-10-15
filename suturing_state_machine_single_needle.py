from utils import *
from enum import Enum

class SuturingState(Enum):
    # go to the designated home position
    # line up to the pose before insertion, calculated by `calculate_desired_entry_pose`
    PREPARE_INSERTION = 0
    # execute a semicircular trajectory to insert the needle
    INSERTION = 1
    # release the needle
    RELEASE_NEEDLE_PSM1 = 2
    # rotate along the circle representing the entire suture throw, further than necessary
    # to pick up the needle, to force the wrist to 'flip'
    OVERROTATE = 3
    # line up to the tip of the needle
    PREPARE_EXTRACTION = 4
    # grasp the needle
    GRASP_NEEDLE_PSM1 = 5
    # execute a semicircular trajectory to extract the needle
    EXTRACTION = 6
    # release the needle again
    RELEASE_NEEDLE_PSM1 = 7
    # pick up the needle
    PICKUP = 8
    # grasp needle again
    GRASP_NEEDLE_PSM2 = 9
    # no more suture throws to execute
    DONE = 11


    RELEASE_NEEDLE_PSM2 = 10
            


class SuturingStateMachine:

    def jaw_fully_open_PSM1(self):
        return True if self.psm1.get_current_jaw_position() >= math.pi / 3 else False 

    def jaw_fully_closed_PSM1(self):
        return True if self.psm1.get_current_jaw_position() <= 0 else False

    def jaw_fully_open_PSM2(self):
            return True if self.psm2.get_current_jaw_position() >= math.pi / 3 else False 

    def jaw_fully_closed_PSM2(self):
        return True if self.psm2.get_current_jaw_position() <= 0 else False



    def _prepare_insertion_state(self):
        if self.circle_pose_PSM1 or self.circle_pose_PSM2 is None:
            self.circle_pose_PSM1 = calculate_desired_entry_pose(self.paired_pts[self.paired_pts_idx], 'PSM1')
            self.circle_pose_PSM2 = calculate_desired_entry_pose(self.paired_pts[self.paired_pts_idx], 'PSM2')

        set_arm_dest(self.psm2, self.tf_world_to_psm2 * self.circle_pose_PSM2)
        

    def _prepare_insertion_next(self):
  
        if arm_pos_reached(self.psm2, self.tf_world_to_psm2 * self.circle_pose_PSM2.p):
            return SuturingState.INSERTION
        else:
            return SuturingState.PREPARE_INSERTION

    
    def _insertion_state(self):
        if self.circular_motion is None:
            self.circular_motion = CircularMotion(self.psm2, self.tf_world_to_psm2, NEEDLE_RADIUS,
                                                  self.paired_pts[self.paired_pts_idx],
                                                  self.circle_pose_PSM2, 0, self.insertion_rads,'PSM2')
        self.circular_motion.step()

    
    def _insertion_next(self):
        if self.circular_motion is not None and self.circular_motion.is_done():
            self.circular_motion = None
            return SuturingState.RELEASE_NEEDLE_PSM2
        else:
            return SuturingState.INSERTION


    def _release_needle_state_PSM2(self):
        if self.psm2.get_desired_jaw_position() <= 0.:
            self.psm2.open_jaw(blocking=False)
            
    def _release_needle_state_PSM1(self):
        if self.psm1.get_desired_jaw_position() <= 0.:
            self.psm1.open_jaw(blocking=False)

    
    def _release_needle_next_PSM2(self):
        if self.jaw_fully_open_PSM2():
            # return SuturingState.OVERROTATE
            return SuturingState.PREPARE_EXTRACTION
        else:
            return SuturingState.RELEASE_NEEDLE_PSM2


    def _overrotate_state(self):
        # an upward motion and a 'over-rotation' to get the gripper in the right pose for 
        # extracting the needle
        overrotation_circle_pose = PyKDL.Frame(self.circle_pose.M, self.circle_pose.p 
                                             + self.circle_pose.M.Inverse() * PyKDL.Vector(0, 0.015, 0))
       
        offset = 0.45
        self.overrotation_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx],
                                                         overrotation_circle_pose,
                                                         self.insertion_rads + np.pi + offset, 
                                                         NEEDLE_RADIUS +0.005)
        set_arm_dest(self.psm2, self.tf_world_to_psm2 * self.overrotation_pose)


    def _overrotate_next(self):
        if arm_pos_reached(self.psm2, self.tf_world_to_psm2 * self.overrotation_pose.p):
            self.overrotation_pose = None
            return SuturingState.PREPARE_EXTRACTION
        else:
            return SuturingState.OVERROTATE

    
    def _prepare_extraction_state(self):
        offset = -0.55
        
        self.psm2.move_joint(PSM_HOME_JOINT_POS, blocking=False)
        self.psm1.open_jaw(blocking=True)

        pickup_rads = self.insertion_rads + np.pi + offset
        opposite_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx], 
                                                self.circle_pose_PSM1, pickup_rads,'PSM1')
        self.prepare_extraction_pose = opposite_pose
        set_arm_dest(self.psm1, self.tf_world_to_psm1 * self.prepare_extraction_pose)
            

    def _prepare_extraction_next(self):
        
        if arm_pos_reached(self.psm1, self.tf_world_to_psm1 * self.prepare_extraction_pose.p):
            self.prepare_extraction_pose = None
            return SuturingState.GRASP_NEEDLE_PSM1
        else:
            return SuturingState.PREPARE_EXTRACTION


    def _grasp_needle_state_PSM1(self):
        if self.psm1.get_desired_jaw_position() >= 0.:
            self.psm1.close_jaw(blocking=True)
            
    def _grasp_needle_state_PSM2(self):
        if self.psm2.get_desired_jaw_position() >= 0.:
            self.psm2.close_jaw(blocking=True)

    
    def _grasp_needle_next_PSM1(self):
        if self.jaw_fully_closed_PSM1():
            return SuturingState.EXTRACTION
        else:
            return SuturingState.GRASP_NEEDLE_PSM1


    def _extraction_state(self):
        if self.circular_motion is None:
            offset = -0.55
            offset2 = 0.15
            
            self.circular_motion = CircularMotion(self.psm1, self.tf_world_to_psm1, NEEDLE_RADIUS,
                                                  self.paired_pts[self.paired_pts_idx],
                                                  self.circle_pose_PSM1, 
                                                  self.insertion_rads + np.pi + offset,
                                                  # TODO: tweak this value
                                                  self.insertion_rads + self.extraction_rads 
                                                  + np.pi + offset2,'PSM1')
        self.circular_motion.step()


    def _extraction_next(self):
        if self.circular_motion is not None and self.circular_motion.is_done():
            self.circle_motion = None
            return SuturingState.RELEASE_NEEDLE_PSM1
        else:
            return SuturingState.EXTRACTION

    
    def _release_needle_next_PSM1(self):
        if self.jaw_fully_open_PSM1():
            return SuturingState.PICKUP
        else:
            return SuturingState.RELEASE_NEEDLE_PSM1


    def _pickup_state(self):
        self.psm1.move_joint(PSM_HOME_JOINT_POS, blocking=False)
        if self.pickup_pose is None:
            
            offset = -0.2
            self.pickup_pose = calculate_circular_pose(self.paired_pts[self.paired_pts_idx], 
                                                       self.circle_pose_PSM2,
                                                       self.insertion_rads + self.extraction_rads +offset,'PSM2')
        set_arm_dest(self.psm2, self.tf_world_to_psm2 * self.pickup_pose)

    def _pickup_next(self):
        if arm_pos_reached(self.psm2, self.tf_world_to_psm2 * self.pickup_pose.p):
            self.pickup_pose = None
            return SuturingState.GRASP_NEEDLE_PSM2
        else:
            return SuturingState.PICKUP


    def _grasp_needle_next_PSM2(self):
        if self.jaw_fully_closed_PSM2():
            self.circle_pose_PSM2 = None
            self.circular_motion = None
            self.paired_pts_idx += 1
            if self.paired_pts_idx < len(self.paired_pts):
                return SuturingState.PREPARE_INSERTION
            else:
                self.psm2.move_joint(PSM_HOME_JOINT_POS, blocking=True)
                return SuturingState.DONE

        else:
            return SuturingState.GRASP_NEEDLE_PSM2



    def is_done(self):
        return self.state == SuturingState.DONE


    def run_once(self):
        if self.state == SuturingState.DONE:
            self.psm2.move_joint(PSM_HOME_JOINT_POS, blocking=True)
            return

        self.state_funs[self.state]()

        if self.state == SuturingState.DONE:
            self.psm2.move_joint(PSM_HOME_JOINT_POS, blocking=True)
            return
        # rospy.loginfo("Executing state {}".format(self.state))
        self.state = self.next_funs[self.state]()


    def __init__(self, psm1,psm2 , tf_world_to_psm1, tf_world_to_psm2, paired_pts, insertion_rads=3.4, extraction_rads=2.4):
        self.psm1 = psm1
        self.psm2 = psm2
        self.tf_world_to_psm1 = tf_world_to_psm1
        self.tf_world_to_psm2 = tf_world_to_psm2
        self.paired_pts = paired_pts
        self.paired_pts_idx = 0
        self.insertion_rads = insertion_rads
        self.extraction_rads = extraction_rads
        
        self.state = SuturingState.PREPARE_INSERTION

        # evidence that i did not spend any time on good code design
        self.circle_pose_PSM1 = None
        self.circle_pose_PSM2 = None
        self.circular_motion = None
        self.overrotation_pose = None
        self.prepare_extraction_pose = None
        self.pickup_pose = None
        self.psm1.move_joint(PSM_HOME_JOINT_POS, blocking=True)
        self.psm2.move_joint(PSM_HOME_JOINT_POS, blocking=True)

        self.state_funs = {
    
            SuturingState.PREPARE_INSERTION : self._prepare_insertion_state,
            SuturingState.INSERTION : self._insertion_state,
            SuturingState.RELEASE_NEEDLE_PSM2 : self._release_needle_state_PSM2,
            SuturingState.RELEASE_NEEDLE_PSM1 : self._release_needle_state_PSM1,
            SuturingState.OVERROTATE : self._overrotate_state,
            SuturingState.PREPARE_EXTRACTION : self._prepare_extraction_state,
            SuturingState.GRASP_NEEDLE_PSM1 : self._grasp_needle_state_PSM1,
            SuturingState.EXTRACTION : self._extraction_state,
            SuturingState.RELEASE_NEEDLE_PSM1 : self._release_needle_state_PSM1,
            SuturingState.PICKUP : self._pickup_state,
            SuturingState.GRASP_NEEDLE_PSM2 : self._grasp_needle_state_PSM2
        }

        self. next_funs = {
           
            SuturingState.PREPARE_INSERTION : self._prepare_insertion_next,
            SuturingState.INSERTION : self._insertion_next,
            SuturingState.RELEASE_NEEDLE_PSM2 : self._release_needle_next_PSM2,
            SuturingState.OVERROTATE : self._overrotate_next,
            SuturingState.PREPARE_EXTRACTION : self._prepare_extraction_next,
            SuturingState.GRASP_NEEDLE_PSM1 :  self._grasp_needle_next_PSM1,
            SuturingState.EXTRACTION : self._extraction_next,
            SuturingState.RELEASE_NEEDLE_PSM1 : self._release_needle_next_PSM1,
            SuturingState.PICKUP : self._pickup_next,
            SuturingState.GRASP_NEEDLE_PSM2 : self._grasp_needle_next_PSM2
        }