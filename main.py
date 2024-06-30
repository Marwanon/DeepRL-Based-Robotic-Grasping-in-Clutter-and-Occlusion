#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
# import utils
# import utils1
import camera_1
import camera_2

def main(args):


    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.41]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
#        workspace_limits_1 = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.7]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    grasp_rewards = args.grasp_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay # Use prioritized experience replay?
    heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only
    push = args.push

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True


    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  is_testing, test_preset_cases, test_preset_file)

    # Initialize trainer
    trainer = Trainer(method, grasp_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu, push)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale, robot.cam_intrinsics_1, robot.cam_pose_1, robot.cam_depth_scale_1) # Save camera intrinsics and pose
#    logger.save_camera_info(robot.cam_intrinsics_1, robot.cam_pose_1, robot.cam_depth_scale_1) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters
    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count_1 = [1,1] if is_testing else [0, 0]
    no_change_count = [1, 1] if  is_testing else [0, 0]
    grasp_failed_iterations = [1]  if  is_testing else [0]
    
    no_change_count = [1, 1] if not is_testing else [0, 0]
    no_change_count_1 = [1,1] if not is_testing else [0, 0]
    grasp_failed_iterations = [2]  if  not is_testing else [0]

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'push_success' : False,
                          'grasp_success' : False}


    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        action_count = 0
        grasp_count = 0
        successful_grasp_count = 0
        grasp_failed_iterations = 0
        no_change_count[0]=0
        no_change_count[1]=0
        no_change_count_1[0]=0
        no_change_count_1[1]=0
        while True:
            if nonlocal_variables['executing_action']:
                action_count += 1
                action_trial = 'action Count: %r' % (action_count)
                print(action_trial)
#                best_push_conf = np.max(push_predictions)
                best_push_conf = np.max(push_predictions)
                best_grasp_conf = np.max(grasp_predictions)
                # print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))
                best_grasp_conf = np.max(grasp_predictions)
                nonlocal_variables['primitive_action'] = 'grasp'
                if push and not is_testing: 
                    if push and no_change_count[0] >= 1 or no_change_count_1[0] >= 1:
                        no_change_count[0] = 0 
                        no_change_count_1[0]=0
                        print('Change not detected for more than one grasp. Running  pushing.')
                        best_push_conf = np.max(push_predictions)
                        print('Primitive confidence scores: %f (push)' % (best_push_conf))
                        nonlocal_variables['primitive_action'] = 'push'
                    elif push  and no_change_count[1] >= 1 or no_change_count_1[1] >= 1:
                        no_change_count[1] = 0 
                        no_change_count_1[1]=0
                        print('Change  not detected for more than one push. Running  pushing.')
                        best_push_conf = np.max(push_predictions)
                        print('Primitive confidence scores: %f (push)' % (best_push_conf))
                        nonlocal_variables['primitive_action'] = 'push'
                    else:
                        print('None above. Running  grasping.')
                        best_grasp_conf = np.max(grasp_predictions)
                        print('Primitive confidence scores: %f (grasp)' % (best_grasp_conf))
                        nonlocal_variables['primitive_action'] = 'grasp'
                elif push and  is_testing:
                    if push and no_change_count[0] >= 1 or no_change_count_1[0] >= 1:
                        no_change_count[0] = 0 
                        no_change_count_1[0]=0
                        print('Change not detected for more than one grasp. Running  pushing.')
                        best_push_conf = np.max(push_predictions)
                        print('Primitive confidence scores: %f (push)' % (best_push_conf))
                        nonlocal_variables['primitive_action'] = 'push'
                    elif push  and no_change_count[1] >= 1 or no_change_count_1[1] >= 1:
                        no_change_count[1] = 0
                        no_change_count_1[1]=0
                        print('Change  not detected for more than one push. Running  pushing.')
                        best_push_conf = np.max(push_predictions)
                        print('Primitive confidence scores: %f (push)' % (best_push_conf))
                        nonlocal_variables['primitive_action'] = 'push'
                    else:
                        print('None above. Running  grasping.')
                        best_grasp_conf = np.max(grasp_predictions)
                        print('Primitive confidence scores: %f (grasp)' % (best_grasp_conf))
                        nonlocal_variables['primitive_action'] = 'grasp'
                # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                    predicted_value = np.max(push_predictions)
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                    predicted_value = np.max(grasp_predictions)
                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)
                
                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                if nonlocal_variables['primitive_action'] == 'grasp' :
                    primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
                elif nonlocal_variables['primitive_action'] == 'push': 
                    primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap_1[best_pix_y][best_pix_x] + workspace_limits[2][0]]
               
                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap_1[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap_1.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap_1.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]- 0.01
                    else:
                        safe_z_position = np.max(local_region) + workspace_limits[2][0]- 0.01
                    primitive_position[2] = safe_z_position
                    print('3D z position:', primitive_position[2])
                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 1 - grasp
                elif nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 0 - push
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
                    grasp_direction_vis = trainer.get_best_grasp_vis(nonlocal_variables['best_pix_ind'], color_heightmap)
                    logger.save_visualizations(trainer.iteration, grasp_direction_vis, 'best_grasp')
                    cv2.imwrite('visualization.best_grasp.png', grasp_direction_vis)
                    if push and nonlocal_variables['primitive_action'] == 'push':
                        push_pred_vis = trainer.get_prediction_vis_1(push_predictions, color_heightmap_1, nonlocal_variables['best_pix_ind'])
                        logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                        cv2.imwrite('visualization.push.png', push_pred_vis)
                        push_direction_vis = trainer.get_best_push_direction_vis(nonlocal_variables['best_pix_ind'], color_heightmap_1)
                        logger.save_visualizations(trainer.iteration, push_direction_vis, 'best_push')
                        cv2.imwrite('visualization.best_push.png', push_direction_vis)
              

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                change_detected = False
                change_detected_1 = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'grasp':
                    grasp_count += 1
                    nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle, workspace_limits)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                    if nonlocal_variables['grasp_success']:
                        successful_grasp_count += 1
                    else:
                        grasp_failed_iterations +=1
                    grasp_rate = float(successful_grasp_count) / float(grasp_count)
                    grasp_str = 'Grasp Count: %r, grasp success rate: %r' % (grasp_count, grasp_rate)
                    print(grasp_str)
                elif nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)

                
                trainer.grasp_success_log.append([int(nonlocal_variables['grasp_success'])])
            

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False

    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        
        color_img_1, depth_img_1 = robot.get_camera_data_1()
        depth_img_1 = depth_img_1 * robot.cam_depth_scale_1 # Apply depth scale from calibration
        color_heightmap_1, depth_heightmap_1 = camera_2.get_heightmap(color_img_1, depth_img_1, robot.cam_intrinsics_1, robot.cam_pose_1, workspace_limits, heightmap_resolution)
        valid_depth_heightmap_1 = depth_heightmap_1.copy()
        valid_depth_heightmap_1[np.isnan(valid_depth_heightmap_1)] = 0
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration
        color_heightmap, depth_heightmap = camera_1.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, color_img_1, depth_img_1, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, color_heightmap_1, valid_depth_heightmap_1, '0')

        # Reset simulation or pause real-world training if table is empty
#        if nonlocal_variables['primitive_action'] == 'push':
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        stuff_count_1 = np.zeros(valid_depth_heightmap_1.shape)
        stuff_count_1[valid_depth_heightmap_1 > 0.02] = 1
        empty_threshold = 200
        if is_sim and is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold and np.sum(stuff_count_1) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 4 or no_change_count_1[0] + no_change_count_1[1] > 4):
            no_change_count = [0, 0]
            no_change_count_1 = [0, 0]
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
#                robot.add_objects_1()
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (np.sum(stuff_count)))
                robot.restart_real()

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

        if not exit_called:

            # Run forward pass with network to get affordances
            push_predictions, grasp_predictions, state_feat_1, state_feat_2 = trainer.forward(color_heightmap, valid_depth_heightmap, color_heightmap_1, valid_depth_heightmap_1, is_volatile=True)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' and 'prev_color_img_1' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 200 
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            print('Change detected-ws1: %r (value: %d)' % (change_detected, change_value))
            
            if change_detected :
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
               if prev_primitive_action == 'push':
                    no_change_count[0] += 1
               elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1
            
            depth_diff_1 = abs(depth_heightmap_1 - prev_depth_heightmap_1)
            depth_diff_1[np.isnan(depth_diff_1)] = 0
            depth_diff_1[depth_diff_1 > 0.3] = 0
            depth_diff_1[depth_diff_1 < 0.01] = 0
            depth_diff_1[depth_diff_1 > 0] = 1
            change_threshold = 200 
            change_value_1 = np.sum(depth_diff_1)
            change_detected_1 = change_value > change_threshold or prev_grasp_success
            print('Change detected-ws2: %r (value: %d)' % (change_detected_1, change_value_1))
            
            if change_detected_1:
                if prev_primitive_action == 'push':
                    no_change_count_1[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count_1[1] = 0
            else:
               if prev_primitive_action == 'push':
                    no_change_count_1[0] += 1
               elif prev_primitive_action == 'grasp':
                    no_change_count_1[1] += 1
            # Compute training labels
            if push:
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, change_detected, 
                                                                         change_detected_1, prev_grasp_predictions, color_heightmap, 
                                                                         valid_depth_heightmap, color_heightmap_1, valid_depth_heightmap_1, 
                                                                         prev_push_success, prev_push_predictions)
            else:
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, change_detected,
                                                                         change_detected_1, prev_grasp_predictions, color_heightmap, 
                                                                         valid_depth_heightmap,color_heightmap_1, valid_depth_heightmap_1, 
                                                                         None, None)
          
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            # Backpropagate
            trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_color_heightmap_1, prev_valid_depth_heightmap_1, prev_primitive_action, prev_best_pix_ind, label_value)

            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    if method == 'reinforcement':
                        sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    if method == 'reinforcement':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))
                if sample_ind.size > 0:

                    # Find sample with highest surprise value
                    if method == 'reinforcement':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
                    # Load sample RGB-D heightmap
                    sample_color_heightmap_1 = cv2.imread(os.path.join(logger.color_heightmaps_1_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap_1 = cv2.cvtColor(sample_color_heightmap_1, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap_1 = cv2.imread(os.path.join(logger.depth_heightmaps_1_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap_1 = sample_depth_heightmap_1.astype(np.float32)/100000
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
                    
                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_push_predictions, sample_grasp_predictions, sample_state_feat_1, sample_state_feat_2 = trainer.forward(sample_color_heightmap, sample_depth_heightmap, sample_color_heightmap_1, sample_depth_heightmap_1, is_volatile=True)

                    # Load next sample RGB-D heightmap
#                    if sample_primitive_action == 'push': #nonlocal_variables['primitive_action'] == 'push':
                    next_sample_color_heightmap_1 = cv2.imread(os.path.join(logger.color_heightmaps_1_directory, '%06d.0.color.png' % (sample_iteration+1)))
                    next_sample_color_heightmap_1 = cv2.cvtColor(next_sample_color_heightmap_1, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap_1 = cv2.imread(os.path.join(logger.depth_heightmaps_1_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
                    next_sample_depth_heightmap_1 = next_sample_depth_heightmap_1.astype(np.float32)/100000
                    next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
                    next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000
#                    sample_push_success = sample_reward_value == 2
#                    sample_grasp_success = sample_reward_value == 1
                    sample_push_success = sample_reward_value == 0.5
                    sample_change_detected = sample_push_success 
                    sample_change_detected_1 = sample_push_success
                    sample_grasp_success = sample_reward_value == 1.0

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
                    trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_color_heightmap_1, sample_depth_heightmap_1, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'push':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                        # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                    elif sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                        # trainer.label_value_log[sample_iteration] = [new_sample_label_value]

                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_color_img_1 = color_img_1.copy()
        prev_depth_img_1 = depth_img_1.copy()
        prev_color_heightmap_1 = color_heightmap_1.copy()
        prev_depth_heightmap_1 = depth_heightmap_1.copy()
        prev_valid_depth_heightmap_1 = valid_depth_heightmap_1.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        if push:
            prev_push_predictions = push_predictions.copy()

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--grasp_rewards', dest='grasp_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)
    parser.add_argument('--push', dest='push', action='store_true', default=False,                                      help='enable pushing of objects')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
