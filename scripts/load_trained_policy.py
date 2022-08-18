"""Example of running A1/Go1 robot with position control.

To run:
python -m src.robots.a1_robot_exercise_example.py
"""
from absl import app
from absl import flags

import ml_collections
import numpy as np
import pybullet
from pybullet_utils import bullet_client
import time
from typing import Tuple

from src.robots import a1, go1
from src.robots import a1_robot, go1_robot
from src.robots.motors import MotorCommand

import torch

flags.DEFINE_bool('use_real_robot', False, 'whether to use real robot.')
FLAGS = flags.FLAGS
DEVICE = torch.device("cpu")
POLICY_PATH = "~/policies/a1_flat.pt"
CFG = {"lin_vel_scale": 2.0,
       "ang_vel_scale": 0.25,
       "dof_pos_scale": 1.0,
       "dof_vel_scale": 0.05,
       "action_scale": 0.25}


def quat_rotate(q, v):
    w, x, y, z = q
    q_vec = q[1:]
    a = v * (2.0 * w ** 2 - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c

def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_vec = q[1:]
    a = v * (2.0 * w ** 2 - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def load_policy():
    return torch.jit.load(POLICY_PATH).to(DEVICE)

def get_action(robot, policy, last_action):
    obs = get_obs(robot, last_action)
    action = policy(obs) * CFG["action_scale"]
    return MotorCommand(desired_position=np.zeros(robot.num_motors),
                      kp=robot.motor_group.kps,
                      desired_velocity=np.zeros(robot.num_motors),
                      kd=robot.motor_group.kds)

def get_obs(robot, last_action):
    lin_vel = robot.base_velocity
    ang_vel = robot.base_angular_velocity_body_frame
    base_q = robot.base_orientation_quat
    projected_gravity = quat_rotate_inverse(base_q, np.array([0, 0, -9.8])))  
    command = np.array([0.2, 0, 0])
    dof_positions = robot.motor_angles
    dof_velocities = robot.motor_velocities
    
    obs = torch.Tensor(np.concatenate([lin_vel * CFG["lin_vel_scale"], 
                                       ang_vel * CFG["ang_vel_scale"],  
                                       projected_gravity, 
                                       command, 
                                       dof_positions * CFG["dof_pos_scale"], 
                                       dof_velocities * CFG["dof_vel_scale"], 
                                       last_action])).to(DEVICE)
    return obs


def get_sim_conf():
    config = ml_collections.ConfigDict()
    config.timestep: float = 0.002
    config.action_repeat: int = 1
    config.reset_time_s: float = 3.
    config.num_solver_iterations: int = 30
    config.init_position: Tuple[float, float, float] = (0., 0., 0.32)
    config.init_rack_position: Tuple[float, float, float] = [0., 0., 1]
    config.on_rack: bool = False
    return config


def main(_):
    if FLAGS.use_real_robot:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        p.setAdditionalSearchPath('src/data')
        p.loadURDF("plane.urdf")
        p.setGravity(0.0, 0.0, -9.8)

    if FLAGS.use_real_robot:
        robot = a1_robot.A1Robot(pybullet_client=p, sim_conf=get_sim_conf())
    else:
        robot = a1.A1(pybullet_client=p, sim_conf=get_sim_conf())
        robot.reset()

    policy = load_policy()
    last_action = np.zeros(12)
    for _ in range(10000):
        action = get_action(robot, policy, last_action)
        last_action = action.copy()
        robot.step(action)
        time.sleep(0.002)
        print(robot.base_orientation_rpy)


if __name__ == "__main__":
    app.run(main)
