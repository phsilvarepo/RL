#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg

def main():
    rospy.init_node('end_effector_position_node', anonymous=True)

    # Create a TF buffer and listener to get transformations from the TF tree
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10)  # Set the rate at which to check for transformations (10 Hz)

    # Define the frame names
    base_frame = 'yumi_base_link'  # Adjust this to the correct base frame of your robot
    end_effector_frame = 'gripper_r_finger_r'  # Adjust this to the correct end effector frame

    while not rospy.is_shutdown():
        try:
            # Get the latest transformation from base_frame to end_effector_frame
            transform = tf_buffer.lookup_transform(base_frame, end_effector_frame, rospy.Time(0))

            # Extract the Cartesian position (translation) from the transform
            translation = transform.transform.translation
            x = translation.x * 100
            y = translation.y * 100
            z = translation.z * 100

            # Extract the quaternion from the transform
            quaternion = transform.transform.rotation
            qx = quaternion.x
            qy = quaternion.y
            qz = quaternion.z
            qw = quaternion.w

            # Print the Cartesian position and quaternion
            rospy.loginfo(f"Cartesian Position of {end_effector_frame} with respect to {base_frame}:")
            rospy.loginfo(f"x: {x}, y: {y}, z: {z}")
            rospy.loginfo(f"Quaternion of {end_effector_frame} with respect to {base_frame}:")
            rospy.loginfo(f"x: {qx}, y: {qy}, z: {qz}, w: {qw}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get transform: {e}")

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
