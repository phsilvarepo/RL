#!/usr/bin/env python3

import time
import numpy as np
import gymnasium as gym

import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import tf

class ReachingYumi(gym.Env):
    def __init__(self):

        # Initialize the ROS node before anything else
        rospy.init_node(self.__class__.__name__)

        # spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(42,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(18,), dtype=np.float32)

        # create publishers
        self.pub_command_joints = rospy.Publisher('/yumi/joint_command', sensor_msgs.msg.JointState, queue_size=1)

        # keep compatibility with libiiwa Python API
        self.robot_state = {"joint_position": np.zeros((18,)),
                            "joint_velocity": np.zeros((18,))}

        # create subscribers
        rospy.Subscriber('/yumi/joint_states', sensor_msgs.msg.JointState, self._callback_joint_states)

        # TF listener setup
        listener = tf.TransformListener()
        listener.waitForTransform('/yumi_base_link', '/gripper_r_finger_r', rospy.Time(), rospy.Duration(0.2))
        (trans, rot) = listener.lookupTransform('/yumi_base_link', '/gripper_r_finger_r', rospy.Time(0))
        self.end_effector_pos = (trans[0] * 100, trans[1] * 100, trans[2] * 100)

        self.object_pos = np.array([0.45, 0.0, 0.0])  
        #rospy.Subscriber('/object_pose', geometry_msgs.msg.Pose, self._callback_object_pose)

        print("Robot connected")

        self.joint_names = [
            'yumi_joint_1_l',
            'yumi_joint_1_r',
            'yumi_joint_2_l',
            'yumi_joint_2_r',
            'yumi_joint_7_l',
            'yumi_joint_7_r',
            'yumi_joint_3_l',
            'yumi_joint_3_r',
            'yumi_joint_4_l',
            'yumi_joint_4_r',
            'yumi_joint_5_l',
            'yumi_joint_5_r',
            'yumi_joint_6_l',
            'yumi_joint_6_r',
            'gripper_l_joint',
            'gripper_l_joint_m',
            'gripper_r_joint',
            'gripper_r_joint_m'
        ]

        self.motion = None
        self.motion_thread = None

        self.dt = 1 / 120.0
        self.action_scale = 7.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 500
        self.robot_dof_speed_scales = np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.1000, 0.1000])
        self.robot_default_dof_pos = np.array([1.1570,  1.1570, -1.0660, -1.0660,  0.4690,  0.4690, -0.1550, -0.1550, -2.2390, -2.2390, -1.8410, -1.8410,  1.0030,  1.0030,  0.0350,  0.0350, 0.0350,  0.0350])
        self.robot_dof_lower_limits = np.array([-2.9409, -2.9409, -2.5045, -2.5045, -2.9409, -2.9409, -2.1555, -2.1555, -5.0615, -5.0615, -1.5359, -1.5359, -3.9968, -3.9968,  0.0000,  0.0000, 0.0000,  0.0000])
        self.robot_dof_upper_limits = np.array([2.9409, 2.9409, 0.7592, 0.7592, 2.9409, 2.9409, 1.3963, 1.3963, 5.0615, 5.0615, 2.4086, 2.4086, 3.9968, 3.9968, 0.0250, 0.0250, 0.0250, 0.0250])

        self.episode_length_buf = 1
        self.observations = np.zeros((42,), dtype=np.float32)

    def _callback_joint_states(self, msg):
        self.robot_state["joint_position"] = np.array(msg.position)
        # Check for NaN values in joint velocities and handle them
        velocity = np.array(msg.velocity)
        if np.isnan(velocity).any():
            rospy.logwarn("Detected NaN values in joint velocities. Replacing NaN with 0.0.")
            velocity = np.nan_to_num(velocity, nan=0.0)  # Replace NaN values with 0.0

        self.robot_state["joint_velocity"] = velocity

    '''def _callback_object_pose(self, msg):
        obj_position = msg.position
        self.object_pos = np.array([obj_position.x, obj_position.y, obj_position.z])   ''' 

    def _get_observation_reward_done(self):
        # observation
        robot_joint_pos = self.robot_state["joint_position"]
        robot_joint_vel = self.robot_state["joint_velocity"]

        listener = tf.TransformListener()
        listener.waitForTransform('/yumi_base_link', '/gripper_r_finger_r', rospy.Time(), rospy.Duration(0.2))
        (trans, rot) = listener.lookupTransform('/yumi_base_link', '/gripper_r_finger_r', rospy.Time(0))
        self.end_effector_pos = np.array([trans[0] * 100, trans[1] * 100, trans[2] * 100])

        object_pos = self.object_pos
        end_effector_pos = self.end_effector_pos

        dof_pos_scaled = 2.0 * (robot_joint_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_joint_vel * self.dof_vel_scale

        to_object = object_pos - end_effector_pos

        #self.observations[0] = self.episode_length_buf / float(self.max_episode_length)
        self.observations[0:18] = dof_pos_scaled
        self.observations[18:36] = dof_vel_scaled
        self.observations[36:39] = to_object
        self.observations[39:42] = object_pos

        # Calculate the distance from the end effector to the object
        distance = np.linalg.norm(end_effector_pos - object_pos)
        # Calculate the reward using an inverse quadratic function
        reward = 1.0 / (1.0 + distance ** 2)

        # done
        done = self.episode_length_buf >= self.max_episode_length - 1
        distance_threshold = distance <= 0.075
        done = done or distance_threshold

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.observations, reward, done

    def reset(self):
        print("Reseting...")

        # go to 1) safe position, 2) random position
        msg = sensor_msgs.msg.JointState()
        msg.name = self.joint_names
        msg.position = self.robot_default_dof_pos.tolist()
        self.pub_command_joints.publish(msg)
    
        time.sleep(3)
        msg.name = self.joint_names

        random_offset = np.random.uniform(-0.125, 0.125, size=self.robot_default_dof_pos.shape)
        joint_positions = self.robot_default_dof_pos + random_offset
        #joint_positions=(self.robot_default_dof_pos + 0.25 * (np.random.rand(18) - 0.5)).tolist()
        msg.position = np.clip(joint_positions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self.pub_command_joints.publish(msg)
        print(msg)

        time.sleep(1)

        self.episode_length_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        return observation, {}

    def step(self, action):
        self.episode_length_buf += 1

        joint_positions = self.robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
        joint_positions = np.clip(joint_positions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        msg = sensor_msgs.msg.JointState()
        msg.position = joint_positions.tolist()
        msg.name = self.joint_names
        self.pub_command_joints.publish(msg)

        # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
        time.sleep(1.0 / 20.0)

        observation, reward, terminated = self._get_observation_reward_done()

        return observation, reward, terminated, False, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass
