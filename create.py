#!/usr/bin/env python

import numpy as np
import os
from robot import Robot


# User options (change me)
# --------------- Setup options ---------------
obj_mesh_dir = os.path.abspath('objects/blocks')
num_obj = 10
random_seed = 1234
workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits_1 = np.asarray([[-0.20, 0.248], [0.302, 0.75], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_Sim = np.asarray([[-0.75, 0.3], [-0.248, 0.75], [-0.0001, 0.4]])
# ---------------------------------------------

# Set random seed
np.random.seed(random_seed)

# Initialize robot simulation
robot = Robot(True, obj_mesh_dir, num_obj, workspace_limits, workspace_limits_1, 
              True, False, None)

test_case_file_name = raw_input("Enter the name of the file: ") # test-10-obj-00.txt

# Fetch object poses
obj_positions, obj_orientations = robot.get_obj_positions_and_orientations()

# Save object information to file
file = open(test_case_file_name, 'w') 
for object_idx in range(robot.num_obj):
    # curr_mesh_file = os.path.join(robot.obj_mesh_dir, robot.mesh_list[robot.obj_mesh_ind[object_idx]]) # Use absolute paths
    curr_mesh_file = os.path.join(robot.mesh_list[robot.obj_mesh_ind[object_idx]])
    file.write('%s %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e\n' % (curr_mesh_file,
                                                                               robot.obj_mesh_color[object_idx][0], robot.obj_mesh_color[object_idx][1], robot.obj_mesh_color[object_idx][2],
                                                                               obj_positions[object_idx][0], obj_positions[object_idx][1], obj_positions[object_idx][2],
                                                                               obj_orientations[object_idx][0], obj_orientations[object_idx][1], obj_orientations[object_idx][2]))
file.close()
