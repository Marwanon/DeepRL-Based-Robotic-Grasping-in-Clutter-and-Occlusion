import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from models import reinforcementlearning
from scipy import ndimage
import matplotlib.pyplot as plt



class Trainer(object):
    def __init__(self, method, grasp_rewards, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file, force_cpu, push=False):

        self.method = method
        self.push = push

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        if self.method == 'reinforcement':
            self.model = reinforcementlearning(self.use_cuda)
            self.grasp_rewards = grasp_rewards
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.clearance_log = []
        self.grasp_success_log = []
        self.change_detected_log = []


    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0],1)
        self.clearance_log = self.clearance_log.tolist()
        self.grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp_success_log.txt'), delimiter=' ')
        self.grasp_success_log = self.grasp_success_log[0:self.iteration]
        self.grasp_success_log.shape = (self.iteration, 1)
        self.grasp_success_log = self.grasp_success_log.tolist()
        self.change_detected_log = np.loadtxt(os.path.join(transitions_directory, 'change-detected.log.txt'), delimiter=' ')
        self.change_detected_log = self.change_detected_log[0:self.iteration]
        self.change_detected_log.shape = (self.iteration, 1)
        self.change_detected_log = self.change_detected_log.tolist()

 
    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, color_heightmap_1, depth_heightmap_1, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        color_heightmap_1_2x = ndimage.zoom(color_heightmap_1, zoom=[2,2,1], order=0)
        depth_heightmap_1_2x = ndimage.zoom(depth_heightmap_1, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2] and color_heightmap_1_2x.shape[0:2] == depth_heightmap_1_2x.shape[0:2])
#        assert(color_heightmap_1_2x.shape[0:2] == depth_heightmap_1_2x.shape[0:2])
        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        
        diag_length_1 = float(color_heightmap_1_2x.shape[0]) * np.sqrt(2)
        diag_length_1 = np.ceil(diag_length_1/32)*32
        padding_width_1 = int((diag_length_1 - color_heightmap_1_2x.shape[0])/2)
        color_heightmap_1_2x_r =  np.pad(color_heightmap_1_2x[:,:,0], padding_width_1, 'constant', constant_values=0)
        color_heightmap_1_2x_r.shape = (color_heightmap_1_2x_r.shape[0], color_heightmap_1_2x_r.shape[1], 1)
        color_heightmap_1_2x_g =  np.pad(color_heightmap_1_2x[:,:,1], padding_width_1, 'constant', constant_values=0)
        color_heightmap_1_2x_g.shape = (color_heightmap_1_2x_g.shape[0], color_heightmap_1_2x_g.shape[1], 1)
        color_heightmap_1_2x_b =  np.pad(color_heightmap_1_2x[:,:,2], padding_width_1, 'constant', constant_values=0)
        color_heightmap_1_2x_b.shape = (color_heightmap_1_2x_b.shape[0], color_heightmap_1_2x_b.shape[1], 1)
        color_heightmap_1_2x = np.concatenate((color_heightmap_1_2x_r, color_heightmap_1_2x_g, color_heightmap_1_2x_b), axis=2)
        depth_heightmap_1_2x =  np.pad(depth_heightmap_1_2x, padding_width_1, 'constant', constant_values=0)
        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        input_color_image_1 = color_heightmap_1_2x.astype(float)/255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c])/image_std[c]
        for c in range(3):
            input_color_image_1[:, :, c] = (input_color_image_1[:, :, c] - image_mean[c])/image_std[c]
        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - image_mean) / image_std
        
        depth_heightmap_1_2x.shape = (depth_heightmap_1_2x.shape[0], depth_heightmap_1_2x.shape[1], 1)
        input_depth_image_1 = (depth_heightmap_1_2x - image_mean) / image_std
        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_color_image_1.shape = (input_color_image_1.shape[0], input_color_image_1.shape[1], input_color_image_1.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_depth_image_1.shape = (input_depth_image_1.shape[0], input_depth_image_1.shape[1], input_depth_image_1.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_color_data_1 = torch.from_numpy(input_color_image_1.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data_1 = torch.from_numpy(input_depth_image_1.astype(np.float32)).permute(3,2,0,1)
        # Pass input data through model
        output_prob, state_feat_1, state_feat_2 = self.model.forward( input_color_data, input_depth_data, input_color_data_1, input_depth_data_1, is_volatile, specific_rotation)

        if self.method == 'reinforcement':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    if self.push:
                        push_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width_1/2):int(color_heightmap_1_2x.shape[0]/2 - padding_width_1/2),int(padding_width_1/2):int(color_heightmap_1_2x.shape[0]/2 - padding_width_1/2)]
                        grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    else:
                        grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    if self.push:
                        push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width_1/2):int(color_heightmap_1_2x.shape[0]/2 - padding_width_1/2),int(padding_width_1/2):int(color_heightmap_1_2x.shape[0]/2 - padding_width_1/2)]), axis=0)
                        grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    else: 
                        grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
            if not self.push:
                push_predictions = None 
        return push_predictions, grasp_predictions, state_feat_1, state_feat_2


    def get_label_value(self, primitive_action, grasp_success, change_detected, 
                        change_detected_1, prev_grasp_predictions, 
                        next_color_heightmap, next_depth_heightmap, next_color_heightmap_1, 
                        next_depth_heightmap_1, push_success, prev_push_predictions ):

        if self.method == 'reinforcement':

            # Compute current reward
            current_reward = 0
            if primitive_action == 'push':
                if change_detected or change_detected_1: 
                    current_reward = 0.5
                elif not change_detected or not change_detected_1: 
                    current_reward = -1.0
            if primitive_action == 'grasp':
                if grasp_success:
                    current_reward = 1.0

            # Compute future reward
            if not change_detected or not change_detected_1 and not grasp_success:
                future_reward = 0
            else:
                if self.push:
                    next_push_predictions, next_grasp_predictions, next_state_feat_1,  next_state_feat_2 = self.forward( next_color_heightmap, next_depth_heightmap, next_color_heightmap_1, next_depth_heightmap_1,  is_volatile=True)
                    future_reward = max(np.max(next_grasp_predictions), np.max(next_push_predictions))
                else:
                    next_push_predictions, next_grasp_predictions,next_state_feat_1,  next_state_feat_2 = self.forward( next_color_heightmap, next_depth_heightmap, next_color_heightmap_1, next_depth_heightmap_1,  is_volatile=True)
                    future_reward = np.max(next_grasp_predictions)
            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if self.push and not self.grasp_rewards:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward

    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, color_heightmap_1, depth_heightmap_1, primitive_action, best_pix_ind, label_value):

        if self.method == 'reinforcement':

            # Compute labels
            label = np.zeros((1,320,320))
            action_area = np.zeros((224,224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224,224))
            tmp_label[action_area > 0] = label_value
            label[0,48:(320-48),48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224,224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'push':

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, state_feat_1, state_feat_2 = self.forward(color_heightmap, depth_heightmap, color_heightmap_1, depth_heightmap_1, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                loss_value = loss_value/2

            elif primitive_action == 'grasp':

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, state_feat_1, state_feat_2 = self.forward( color_heightmap, depth_heightmap, color_heightmap_1, depth_heightmap_1, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                push_predictions, grasp_predictions, state_feat_1, state_feat_2 = self.forward( color_heightmap, depth_heightmap, color_heightmap_1, depth_heightmap_1, is_volatile=False, specific_rotation=opposite_rotate_idx)

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas
    
    def get_prediction_vis_1(self, predictions, color_heightmap_1, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap_1, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas 
    
    def get_grasp_vis(self, grasp_predictions, color_heightmap, best_pix_ind):
        grasp_canvas = color_heightmap
        x = 0
        while x < grasp_predictions.shape[2]:
            y = 0
            while y < grasp_predictions.shape[1]:
                angle_idx = np.argmax(grasp_predictions[:, y, x])
                angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
                quality = np.max(grasp_predictions[:, y, x])
                
                color = (0, 0, (quality*255).astype(np.uint8))
                cv2.circle(grasp_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                y+=10
            x+=10

        plt.figure()
        plt.imshow(grasp_canvas)
        plt.show()
        return grasp_canvas


    def get_best_grasp_vis(self, best_pix_ind, color_heightmap):
        grasp_canvas = color_heightmap
        angle_idx = best_pix_ind[0]
        angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
        cv2.circle(grasp_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 4, (0,0,255), 1)

        return grasp_canvas
    
    def get_push_direction_vis(self, push_predictions, color_heightmap_1):
        push_direction_canvas = color_heightmap_1
        x = 0
        while x < push_predictions.shape[2]:
            y = 0
            while y < push_predictions.shape[1]:
                angle_idx = np.argmax(push_predictions[:, y, x])
                angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
                start_point = (x, y)
                end_point = (int(x + 10*np.cos(angel)), int(y + 10*np.sin(angel)))
                quality = np.max(push_predictions[:, y, x])
                
                color = (0, 0, (quality*255).astype(np.uint8))
                cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
                y+=10
            x+=10

        plt.figure()
        plt.imshow(push_direction_canvas)
        plt.show()
        return push_direction_canvas


    def get_best_push_direction_vis(self, best_pix_ind, color_heightmap_1):
        push_direction_canvas = color_heightmap_1
        angle_idx = best_pix_ind[0]
        angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
        start_point = (best_pix_ind[2], best_pix_ind[1])
        end_point = (int(best_pix_ind[2] + 20*np.cos(angel)), int(best_pix_ind[1] + 20*np.sin(angel)))
        cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
        cv2.circle(push_direction_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 4, (0,0,255), 1)

        return push_direction_canvas
    
