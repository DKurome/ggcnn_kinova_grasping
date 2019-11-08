#! /usr/bin/env python

import rospy
import tf.transformations as tft

import numpy as np

import kinova_msgs.msg
import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg

from helpers.gripper_action_client import set_finger_positions
from helpers.position_action_client import position_client, move_to_position
from helpers.transforms import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from helpers.covariance import generate_cartesian_covariance

MOVING = False  # 标记机器人是否在速度控制下移动。
CURR_Z = 0  # 当前末端执行器的z高度。

def robot_wrench_callback(msg):
    # 监视扳手以消除碰撞时的运动。
    global MOVING
    if MOVING and msg.wrench.force.z < -2.0:
        MOVING = False
        rospy.logerr('Force Detected. Stopping.')


def robot_position_callback(msg):
    # 监视机器人位置。
    global CURR_Z
    CURR_Z = msg.pose.position.z


def move_to_pose(pose):
    #包装器移动到位置。
    p = pose.position
    o = pose.orientation
    move_to_position([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])


def execute_grasp():
    # 执行抓取。
    global MOVING
    global CURR_Z
    global start_force_srv
    global stop_force_srv

    # 得到位置。
    msg = rospy.wait_for_message('/ggcnn/out/command', std_msgs.msg.Float32MultiArray)
    d = list(msg.data)

    # 计算抓爪宽度。
    grip_width = d[4]
    # 将以像素为单位的宽度转换为毫米。
    # 0.07是从末端执行器（CURR_Z）到摄像机的距离。
    #对于实感，每个像素约0.1度。
    g_width = 2 * ((CURR_Z + 0.07)) * np.tan(0.1 * grip_width / 2.0 / 180.0 * np.pi) * 1000
    # 转换为电机位置。
    g = min((1 - (min(g_width, 70)/70)) * (6800-4000) + 4000, 5500)
    set_finger_positions([g, g])

    rospy.sleep(0.5)

    # 握把的姿势（仅位置）在相机框架中。
    gp = geometry_msgs.msg.Pose()
    gp.position.x = d[0]
    gp.position.y = d[1]
    gp.position.z = d[2]
    gp.orientation.w = 1

    # 转换为基本框架，并添加角度（以确保平面抓地力，不保证相机垂直）。
    gp_base = convert_pose(gp, 'camera_depth_optical_frame', 'm1n6s200_link_base')

    q = tft.quaternion_from_euler(np.pi, 0, d[3])
    gp_base.orientation.x = q[0]
    gp_base.orientation.y = q[1]
    gp_base.orientation.z = q[2]
    gp_base.orientation.w = q[3]

    publish_pose_as_transform(gp_base, 'm1n6s200_link_base', 'G', 0.5)

    # 初始姿势的偏移量。
    initial_offset = 0.20
    gp_base.position.z += initial_offset

    # 禁用力控制，使机器人更加精确。
    stop_force_srv.call(kinova_msgs.srv.StopRequest())

    move_to_pose(gp_base)
    rospy.sleep(0.1)

    # 启动力控制，有助于防止不良碰撞。
    start_force_srv.call(kinova_msgs.srv.StartRequest())

    rospy.sleep(0.25)

    # 重设位置
    gp_base.position.z -= initial_offset

    # 标记以检查碰撞。
    MOVING = True

    # 为控制器生成非线性。
    cart_cov = generate_cartesian_covariance(0)

    # 在速度控制下直线向下移动。
    velo_pub = rospy.Publisher('/m1n6s200_driver/in/cartesian_velocity', kinova_msgs.msg.PoseVelocity, queue_size=1)
    while MOVING and CURR_Z - 0.02 > gp_base.position.z:
        dz = gp_base.position.z - CURR_Z - 0.03   # Offset by a few cm for the fingertips.
        MAX_VELO_Z = 0.08
        dz = max(min(dz, MAX_VELO_Z), -1.0*MAX_VELO_Z)

        v = np.array([0, 0, dz])
        vc = list(np.dot(v, cart_cov)) + [0, 0, 0]
        velo_pub.publish(kinova_msgs.msg.PoseVelocity(*vc))
        rospy.sleep(1/100.0)

    MOVING = False

    # 闭合手指。
    rospy.sleep(0.1)
    set_finger_positions([8000, 8000])
    rospy.sleep(0.5)

    # 移回初始位置。
    gp_base.position.z += initial_offset
    gp_base.orientation.x = 1
    gp_base.orientation.y = 0
    gp_base.orientation.z = 0
    gp_base.orientation.w = 0
    move_to_pose(gp_base)

    stop_force_srv.call(kinova_msgs.srv.StopRequest())

    return


if __name__ == '__main__':
    rospy.init_node('ggcnn_open_loop_grasp')

    # 机器人显示器。
    wrench_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_wrench', geometry_msgs.msg.WrenchStamped, robot_wrench_callback, queue_size=1)
    position_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', geometry_msgs.msg.PoseStamped, robot_position_callback, queue_size=1)

    # https://github.com/dougsm/rosbag_recording_services
    # start_record_srv = rospy.ServiceProxy('/data_recording/start_recording', std_srvs.srv.Trigger)
    # stop_record_srv = rospy.ServiceProxy('/data_recording/stop_recording', std_srvs.srv.Trigger)

    # 启用/禁用力控制。
    start_force_srv = rospy.ServiceProxy('/m1n6s200_driver/in/start_force_control', kinova_msgs.srv.Start)
    stop_force_srv = rospy.ServiceProxy('/m1n6s200_driver/in/stop_force_control', kinova_msgs.srv.Stop)

    # 起始位置。
    move_to_position([0, -0.38, 0.25], [0.99, 0, 0, np.sqrt(1-0.99**2)])

    while not rospy.is_shutdown():

        rospy.sleep(0.5)
        set_finger_positions([0, 0])
        rospy.sleep(0.5)

        raw_input('Press Enter to Start.')

        # start_record_srv(std_srvs.srv.TriggerRequest())
        rospy.sleep(0.5)
        execute_grasp()
        move_to_position([0, -0.38, 0.25], [0.99, 0, 0, np.sqrt(1-0.99**2)])
        rospy.sleep(0.5)
        # stop_record_srv(std_srvs.srv.TriggerRequest())

        raw_input('Press Enter to Complete')
