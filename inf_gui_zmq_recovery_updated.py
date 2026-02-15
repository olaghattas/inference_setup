import sys 
import torch
import os
import dill
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import cv2
import time
import threading
from tkinter import *
import zmq

# Utils Imports
sys.path.append("/home/franka_deoxys/fr3")
from util_eval import RobotStateRawObsDictGenerator, FrameStackForTrans

# Diffusion Policy Imports
sys.path.append("/home/franka_deoxys/diffusion_policy")
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

# Deoxys and Franka Interface Imports
sys.path.append("/home/franka_deoxys/deoxys_control/deoxys")
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import  YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info


# --- ASYNCHRONOUS NETWORK WATCHDOG ---
class NetworkWatchdog:
    def __init__(self, monitor_ip='192.168.1.48', data_port=5540, cmd_port=5546):
        self.context = zmq.Context()

        # Socket 1: DATA OUT (Robot -> Monitor)
        # We use PUB (Publish) so we can just blast data without waiting
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{monitor_ip}:{data_port}")

        # Socket 2: COMMAND IN (Monitor -> Robot)
        # We use SUB (Subscribe) to listen for the "STOP" signal
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{monitor_ip}:{cmd_port}")
        self.sub_socket.subscribe(b"") # Listen to everything

        # IMPORTANT: Make the SUB socket ignore old messages (Conflate)
        # We only care about the NEWEST command.
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)

    def compress_image(self, img):
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        # JPG Compression to keep latency low (essential for streaming)
        _, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return encoded
    
    def send_reset_joints(self):
        """Non-blocking send that robot joints were reset"""
        try:
            payload = {
                'event': 'robot_reset',
                'timestamp': time.time()
            }

            # Send immediately (Fire and Forget)
            self.pub_socket.send_pyobj(payload)

        except Exception as e:
            print(f"Watchdog Send Error: {e}")

    def send_continue_monitor(self):
        """Non-blocking send monitor should continue"""
        try:
            payload = {
                'event': 'robot_continue',
                'timestamp': time.time()
            }

            # Send immediately (Fire and Forget)
            self.pub_socket.send_pyobj(payload)

        except Exception as e:
            print(f"Watchdog Send Error: {e}")

    def send_state(self, obs_dict):
        """Non-blocking send of current state"""
        try:
            # Prepare payload
            # print("obs_dict: ", obs_dict)
            payload = {
                'q': obs_dict['joint_states'],
                'ee_pos': obs_dict['ee_states'],
                'gripper_state': obs_dict['gripper_states'],
                'timestamp': time.time()
            }

            # Compress Images
            if 'agentview_rgb' in obs_dict:
                img0 = obs_dict['agentview_rgb']
                if img0.shape[0] == 3: img0 = img0.transpose(1, 2, 0)
                payload['cam0_jpg'] = self.compress_image(img0)

            if 'eye_in_hand_rgb' in obs_dict:
                img1 = obs_dict['eye_in_hand_rgb']
                if img1.shape[0] == 3: img1 = img1.transpose(1, 2, 0)
                payload['cam1_jpg'] = self.compress_image(img1)


            # Send immediately (Fire and Forget)
            self.pub_socket.send_pyobj(payload)

        except Exception as e:
            print(f"Watchdog Send Error: {e}")

    def poll_command(self):
        """
        Non-blocking poll for the latest supervisor command.
        Returns:
            dict | None
        """
        try:
            msg = self.sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
            return msg
        except zmq.Again:
            return None
        except Exception as e:
            print(f"Watchdog Receive Error: {e}")
            return None

    def should_stop(self):
        """
        Checks if a STOP command has been received.
        Returns True if we should stop.
        """
        try:
            # NOBLOCK is the key here. It checks the socket and returns immediately.
            # If no message is waiting, it raises zmq.Again
            msg = self.sub_socket.recv_pyobj(flags=zmq.NOBLOCK)

            if msg.get('command') == 'STOP':
                print("!!! EMERGENCY STOP RECEIVED FROM MONITOR !!!")
                return True
            elif msg.get('command') == 'TASK_DONE':
                print("!!! TASK_DONE RECEIVED FROM MONITOR !!!")
                return True

        except zmq.Again:
            # No message received, everything is fine
            return False
        except Exception as e:
            print(f"Watchdog Receive Error: {e}")

        return False

# --- ROBOT CLASS INTEGRATION ---
class DiffusionPolicyInference():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = robot_config_parse_args()
        
        # Load Robot and Controller Configurations
        self.robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg))
        self.controller_cfg = YamlConfig(
            os.path.join(config_root, args.controller_cfg)
        ).as_easydict()
        self.controller_type = args.controller_type
        self.raw_obs_dict_generator = RobotStateRawObsDictGenerator()

        self.reset_joint_positions = [-0.001923227275113637,
                                        -0.34407008431508795,
                                        -0.0812338482195611,
                                        -2.1801162407414107,
                                        -0.012552774313474605,
                                        1.7719391528668744,
                                        0.8531548323184135]
        
        # Setup Camera Interfaces
        camera_ids = [0, 1]
        self.camera_ids = camera_ids
        self.cr_interfaces = {}

        for camera_id in camera_ids:
            camera_ref=f"rs_{camera_id}"
            assert_camera_ref_convention(camera_ref)
            camera_info = get_camera_info(camera_ref)
            print('---------****-----------')
            print(camera_info)
            print('--------------------------')
       
            cr_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False, redis_host='127.0.0.1')
            cr_interface.start()
            self.cr_interfaces[camera_id] = cr_interface
            
        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        self.keys_select = ['agentview_rgb', 'joint_states', 'ee_states', 'eye_in_hand_rgb', 'gripper_states']

        ## load policy
        # checkpoint = "/home/franka_deoxys/ola_inference/epoch_1100_20260120_102355.ckpt"
        # checkpoint = "/home/franka_deoxys/ola_inference/epoch_1400_20260206_211419_train_loss_0.0018.ckpt"
        # checkpoint = "/home/franka_deoxys/ola_inference/epoch_1600_20260207_132921_train_loss_0.0021.ckpt"
        checkpoint = "/home/franka_deoxys/ola_inference/epoch_1700_20260207_213806_train_loss_0.0025.ckpt"

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        ## not sure 
        # torch.manual_seed(42)
        # np.random.seed(42)
        workspace = TrainDiffusionUnetHybridWorkspace(cfg, output_dir=None)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.to(self.device)
        policy.eval()
        print("Diffusion policy loaded and set to eval mode.")
        self.policy = policy

        # Flags for control
        self.stop_inf = False
        self.max_steps = 10000

        # Initialize Watchdog
        # Change IP to the computer running the monitor
        self.watchdog = NetworkWatchdog(monitor_ip='192.168.1.48')

        self.in_recovery = False
        self.rollout_thread = None

        self.shutdown_event = threading.Event()
        self.rollout_active = threading.Event()
        self.command_lock = threading.Lock()

        self.supervisor_thread = threading.Thread(
            target=self.supervisor_listener,
            daemon=True
        )
        self.supervisor_thread.start()


    def supervisor_listener(self):
        print("Supervisor listener started")

        while not self.shutdown_event.is_set():
            msg = self.watchdog.poll_command()
            # print(f"Received command: {msg}")
            if msg is not None:
                print(f"Received command: {msg}")
                with self.command_lock:
                    self.handle_supervisor_command(msg)

            time.sleep(0.01)  # 100 Hz, low CPU

    ## get states
    def get_imgs(self, use_depth=False):
        data = {}
        for camera_id in self.camera_ids:
            img_info = self.cr_interfaces[camera_id].get_img_info()
            data[f"camera_{camera_id}"]=img_info

            imgs = self.cr_interfaces[camera_id].get_img()
        
            color_img = imgs["color"][..., ::-1]
            size = None ## already the same size as what it should be

            color_img = cv2.resize(color_img, size, fx=0.5, fy=0.5)

            data[f"camera_{camera_id}_color"] = color_img

            if use_depth:
                depth_img = imgs["depth"]
                depth_img = cv2.resize(depth_img, size)
                data[f"camera_{camera_id}_depth"] = depth_img
        return data

    def get_current_obs(self):
        last_state = self.robot_interface._state_buffer[-1]
        # print(last_state)
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        
        obs_dict = self.raw_obs_dict_generator.get_raw_obs_dict(
        {"last_state": last_state, "last_gripper_state": last_gripper_state})
        data = self.get_imgs()
        
        agentview_rgb = data['camera_0_color']
        eye_in_hand_rgb = data['camera_1_color']

       
        obs_dict['agentview_rgb'] = agentview_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        obs_dict['eye_in_hand_rgb'] = eye_in_hand_rgb.transpose(2,  0, 1).astype(np.float32)/255.0
        return obs_dict

    ## helper functions
    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction    
        
    def show_current_images(self):
        data = self.get_imgs()
        print('keys: ', data.keys())

        agentview_rgb = data['camera_0_color']
        eye_in_hand_rgb = data['camera_1_color']
        plt.subplot(112)
        plt.imshow(agentview_rgb[:,:,::-1])
        plt.subplot(122)
        plt.imshow(eye_in_hand_rgb[:,:,::-1])

    def print_state_shapes(self):
        obs_dict = self.get_current_obs()
        for key in obs_dict.keys():
            obs_dict[key]=obs_dict[key][None]
            print(key, obs_dict[key].shape)
            
    ## features
    def set_gripper(self, open=True):
        d=-1. if open else 1.0
        action_close = np.array([ 0.,  0., -0.,  0.,  0., -0., d])
        self.robot_interface.control(
            controller_type=self.controller_type,
            action=action_close,
            controller_cfg=self.controller_cfg,
        )

    def move_joints(self, joint_angles, tol=0.01, timeout=10.0):
        reset_joints_to(self.robot_interface, joint_angles)
        # self.set_gripper(open=True)

        start = time.time()
        while True:
            current_state = self.robot_interface._state_buffer[-1]
            current = np.array(current_state.q)
            error = np.max(np.abs(
                np.array(current) - np.array(joint_angles)
            ))

            if error < tol:
                break

            if time.time() - start > timeout:
                raise TimeoutError("Joint reset timed out")

            time.sleep(0.05)  # don't hammer the controller

        print("Joints reset confirmed")
        
    def reset_joints(self, tol=0.01, timeout=10.0):
        self.watchdog.send_reset_joints()
        reset_joints_to(self.robot_interface, self.reset_joint_positions)
        self.set_gripper(open=True)

        start = time.time()
        while True:
            current_state = self.robot_interface._state_buffer[-1]
            current = np.array(current_state.q)
            error = np.max(np.abs(
                np.array(current) - np.array(self.reset_joint_positions)
            ))

            if error < tol:
                break

            if time.time() - start > timeout:
                raise TimeoutError("Joint reset timed out")

            time.sleep(0.05)  # don't hammer the controller

        print("Joints reset confirmed")
        

    def reset_policy_internal(self):
        self.policy.reset()

    def close_robot_interface(self):
        self.robot_interface.close()
        
    def stop_inf_func(self):
        self.stop_inf = True
        
    # Inference logic
    def predict_action(self, obs):
        """ 
        obs: 2x...
        """
        np_obs_dict = {key:obs[key] for key in self.keys_select}
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=self.device))

        for key in obs_dict.keys():
            obs_dict[key] = obs_dict[key].unsqueeze(0) 

        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
         

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

        # step env
        env_action = np_action_dict['action']
        env_action = self.undo_transform_action(env_action)

        env_action = env_action.squeeze()
        return env_action

    def continue_after_recovery(self):
        """
        Called after RESET / GOTO / manual intervention.
        Safely restarts policy rollout.
        """
        print("continue_after_recovery")
        if self.in_recovery:
            print("Resuming policy rollout from recovery")
            self.stop_inf = False
            self.in_recovery = False
            self.reset_policy_internal()

            self.rollout_thread = threading.Thread(
                target=self.rollout_policy,
                daemon=True
            )
            self.rollout_thread.start()

            
    def handle_supervisor_command(self, msg):
        print("handle_supervisor_command")
        cmd = msg.get("command")

        if cmd == "STOP":
            print("Supervisor STOP received")
            self.stop_inf = True
            self.in_recovery = True

        elif cmd == "RESET_POSE":
            print("Supervisor RESET_POSE received")
            self.stop_inf = True
            self.in_recovery = True
            self.reset_joints()

        elif cmd == "GOTO_POSE":
            print("Supervisor GOTO_POSE received")
            joints = msg.get("joint_angles", None)

            if joints is None or len(joints) != 7:
                print("Invalid GOTO_POSE command")
                return

            self.stop_inf = True
            self.in_recovery = True
            self.move_joints(joints)
        

        elif cmd == "CONTINUE":
            print("Supervisor CONTINUE received")
            if self.in_recovery:
                print("Supervisor CONTINUE received")
                self.continue_after_recovery()

        elif cmd == "TASK_DONE":
            print("Supervisor TASK_DONE received")
            self.stop_inf = True

    def is_robot_moving(self, joint_threshold=1e-2, gripper_threshold=1e-2):
        current = np.array(self.robot_interface._state_buffer[-1].q)
        current_gripper = self.robot_interface._gripper_state_buffer[-1].width
        if not hasattr(self, "_prev_q"):
            self._prev_q = current
            self._prev_gripper = current_gripper
            return False

        joint_diff = np.linalg.norm(current - self._prev_q)
        gripper_diff = abs(current_gripper - self._prev_gripper)
        print("joint_diff: ", joint_diff)
        moving = (joint_diff > joint_threshold) or (gripper_diff > gripper_threshold)
       
        print(f"self._prev_gripper {self._prev_gripper}")
        print(f"current_gripper {current_gripper}")
        print(f"joint_diff > joint_threshold {joint_diff > joint_threshold}")
        print(f"gripper_diff > gripper_threshold {gripper_diff > gripper_threshold}")
        print(f"moving {moving}")
        self._prev_q = current
        self._prev_gripper = current_gripper

        return moving


    def rollout_policy(self, n_obs_steps=2):
        # imgs = []
        print("Starting Policy Rollout...")
        self.stop_inf = False

        framestacker = FrameStackForTrans(n_obs_steps)
        self.obs_dict = self.get_current_obs()
        self.obs_dict = framestacker.reset(self.obs_dict)
        step = 0
        
        
        while not self.stop_inf and step < self.max_steps:
            
            if len(self.robot_interface._state_buffer) == 0:
                continue
            
            self.obs_dict=self.get_current_obs()

            # 1. Check if Monitor sent "STOP" recently
            if self.stop_inf:
                break
    
            # 2. Send current state to Monitor (Fire and Forget)
            if self.is_robot_moving():
                print("moving")
                self.watchdog.send_state(self.obs_dict)

            # self.watchdog.send_state(self.obs_dict)
            
            obs = framestacker.add_new_obs(self.obs_dict)
            action_pred = self.predict_action(obs)
            # print(action_pred)
            # print(f'actions: {action[0]:.3f} {action[1]:.3f} {action[2]:.3f} {action[-1]:.3f}')
            
            for action in action_pred[:4]:
                if self.stop_inf:
                    break

                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg,
                )

            step = step + 1

        if self.stop_inf:
            print("Rollout stopped → entering recovery mode")
            self.in_recovery = True
            
            
def main():
    # 1. Initialize Robot Interface First
    print("Initializing Robot Interface...")
    dif_inf = DiffusionPolicyInference()
    print("Robot Interface Initialized.")

    # 2. Setup GUI
    root = Tk()
    root.title("Robot Control Panel")
    root.geometry('350x850')
    

    # Entry for predicate vector input
    Label(root, text="Joints (comma-separated):").pack(padx=10, pady=5)
    entry = Entry(root, width=50)
    entry.pack(padx=10, pady=5)

    # Status label
    status_label = Label(root, text="Status: Idle")
    status_label.pack(padx=10, pady=5)

    def parse_joint_angles():
        """
        Parse comma-separated joint angles from the entry box
        and call reset_joints_to.
        """
        val = entry.get().strip()

        try:
            # Split by comma
            parts = [p.strip() for p in val.split(",")]

            # Convert to floats
            joint_angles = [float(p) for p in parts]

            EXPECTED_JOINTS = 7
            if len(joint_angles) != EXPECTED_JOINTS:
                raise ValueError(
                    f"Expected {EXPECTED_JOINTS} joint angles, got {len(joint_angles)}"
                )

            print("[Backdoor] Parsed joint angles:", joint_angles)
            print(f" joint_angles {dif_inf.reset_joint_positions}")

            print("[Backdoor] Parsed joint angles:", type(joint_angles))
            print(f" joint_angles {type(dif_inf.reset_joint_positions)}")
            # -0.0019, -0.344, -0.08, -2.1, -0.012, 1.77, 0.85
            # Run reset in a thread to avoid freezing GUI
            t = threading.Thread(
                target=dif_inf.move_joints,
                args=(joint_angles,),
                daemon=True
            )
            t.start()

            status_label.config(text="Joint reset command sent.")

        except ValueError as e:
            status_label.config(text=f"Invalid joint input: {e}")
        except Exception as e:
            status_label.config(text=f"Error: {e}")

        status_label.config(text=f"Printed predicates in console")

    # State variable to track if thread is alive
    is_running = False

    # --- Button Callbacks ---
    btn_reset_joints_entry = Button(
        root,
        text="Reset Joints (From Entry)",
        command=parse_joint_angles,
        bg="orange",
        fg="black",
        activebackground="darkorange"
    )
    btn_reset_joints_entry.pack(pady=15)


    def btn_reset_pos_clicked():
        """Runs reset_joints in a thread to avoid freezing GUI"""
        print("Requesting Joint Reset...")
        # We run this in a thread because moving the robot takes time
        t = threading.Thread(target=dif_inf.reset_joints, daemon=True)
        t.start()


    def btn_continue_clicked():
        """Sends CONTINUE signal to monitor"""
        print("Sending CONTINUE to monitor...")
        dif_inf.watchdog.send_continue_monitor()

    def btn_reset_pol_clicked():
        """Resets the internal policy state"""
        print("Resetting Policy State...")
        dif_inf.reset_policy_internal()
        print("Policy State Reset.")

    def btn_open_grip_clicked():
        """Open gripper"""
        dif_inf.set_gripper(open=True)
        print("Open gripper.")


    def btn_close_grip_clicked():
        """Close gripper"""
        dif_inf.set_gripper(open=False)
        print("Close gripper.")


    def btn_start_pol_clicked():
        """Starts the rollout loop in a separate thread"""
        global is_running
        # Check if thread is already active to prevent double-click issues
        # Ensure stop flag is False before starting
        dif_inf.stop_inf = False

        print("Starting Policy Thread...")
        t = threading.Thread(target=dif_inf.rollout_policy, daemon=True)
        t.start()

    def btn_stop_pol_clicked():
        """Sets the flag to stop the loop"""
        print("Stopping Policy...")
        dif_inf.stop_inf_func()

    # --- Widget Layout ---

    # Reset Position
    btn_reset_pos = Button(root, text="Reset Position",
                           command=btn_reset_pos_clicked,
                           activebackground="grey", activeforeground="white")
    btn_reset_pos.pack(pady=15)

    btn_open_gripper = Button(root, text="Open Gripper",
                           command=btn_open_grip_clicked,
                           activebackground="grey", activeforeground="white")
    btn_open_gripper.pack(pady=15)

    btn_close_gripper = Button(root, text="Close Gripper",
                           command=btn_close_grip_clicked,
                           activebackground="grey", activeforeground="white")
    btn_close_gripper.pack(pady=15)

    # Reset Policy
    btn_reset_pol = Button(root, text="Reset Policy",
                           command=btn_reset_pol_clicked,
                           activebackground="grey", activeforeground="white")
    btn_reset_pol.pack(pady=15)

    # Start Policy
    btn_start_pol = Button(root, text="START Policy",
                           command=btn_start_pol_clicked,
                           bg="green", fg="white", # Make it green by default
                           activebackground="darkgreen", activeforeground="white")
    btn_start_pol.pack(pady=15)

    # Stop Policy
    btn_stop_pol = Button(root, text="STOP Policy",
                          command=btn_stop_pol_clicked,
                          bg="red", fg="white", # Make it red by default
                          activebackground="darkred", activeforeground="white")
    btn_stop_pol.pack(pady=15)

    # Continue Monitor
    btn_continue_pol = Button(root, text="CONTINUE",
        command=btn_continue_clicked,
        bg="blue", fg="white",
        activebackground="darkblue", activeforeground="white")
    btn_continue_pol.pack(pady=15)

    # Handle Window Close Cleanly
    def on_closing():
        print("Closing application...")
        dif_inf.stop_inf_func() # Stop robot
        dif_inf.close_robot_interface() # Close connections
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start Main Loop
    root.mainloop()

if __name__ == '__main__':
    main()
