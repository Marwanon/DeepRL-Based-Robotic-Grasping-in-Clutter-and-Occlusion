"""
Created on Fri Jan 22 17:49:37 2021

@author: marwan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

class FeatureTunkGrasp(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureTunkGrasp, self).__init__()
        self.color_extractor = BasicBlock(3, 3)
        self.depth_extractor = BasicBlock(1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.dense121 = torchvision.models.densenet.densenet121(pretrained=pretrained).features
        self.dense121.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, color, depth):
        return self.dense121(torch.cat((self.color_extractor(color), self.depth_extractor(depth)), dim=1))


class FeatureTunkPush(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureTunkPush, self).__init__()
        self.color_extractor = BasicBlock(3, 3)
        self.depth_extractor = BasicBlock(1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.dense121 = torchvision.models.densenet.densenet121(pretrained=pretrained).features
        self.dense121.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, color, depth):
        return self.dense121(torch.cat((self.color_extractor(color), self.depth_extractor(depth)), dim=1))


class reinforcementlearning(nn.Module):
    def __init__(self, use_cuda):
        super(reinforcementlearning, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.feature_tunk_grasp = FeatureTunkGrasp()
        self.feature_tunk_push = FeatureTunkPush()

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(1024)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(128)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm2', nn.BatchNorm2d(32)),
            ('grasp-relu2', nn.ReLU(inplace=True)),
            ('grasp-conv2', nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))
        ]))

        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(1024)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(128)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=False)),
            ('push-norm2', nn.BatchNorm2d(32)),
            ('push-relu2', nn.ReLU(inplace=True)),
            ('push-conv2', nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' in m[0] or 'push-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat_grasp = []
        self.interm_feat_push = []
        self.output_prob = []

    def forward(self, colorgrasp, depthgrasp, colorpush, depthpush, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before_grasp = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), colorgrasp.size(), align_corners=True)
                        flow_grid_before_push = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), colorpush.size(), align_corners=True)
                    else:
                        flow_grid_before_grasp = F.affine_grid(Variable(affine_mat_before, requires_grad=False), colorgrasp.size(), align_corners=True)
                        flow_grid_before_push = F.affine_grid(Variable(affine_mat_before, requires_grad=False), colorpush.size(), align_corners=True)

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_colorgrasp = F.grid_sample(Variable(colorgrasp).cuda(), flow_grid_before_grasp, align_corners=True)
                        rotate_depthgrasp = F.grid_sample(Variable(depthgrasp).cuda(), flow_grid_before_grasp, align_corners=True)
                        rotate_colorpush = F.grid_sample(Variable(colorpush).cuda(), flow_grid_before_push, align_corners=True)
                        rotate_depthpush = F.grid_sample(Variable(depthpush).cuda(), flow_grid_before_push, align_corners=True)
                    else:
                        rotate_colorgrasp = F.grid_sample(Variable(colorgrasp), flow_grid_before_grasp, align_corners=True)
                        rotate_depthgrasp = F.grid_sample(Variable(depthgrasp), flow_grid_before_grasp, align_corners=True)
                        rotate_colorpush = F.grid_sample(Variable(colorpush), flow_grid_before_push, align_corners=True)
                        rotate_depthpush = F.grid_sample(Variable(depthpush), flow_grid_before_push, align_corners=True)

                    # Compute intermediate features
                    interm_feat_grasp = self.feature_tunk_grasp(rotate_colorgrasp, rotate_depthgrasp)
                    interm_feat_push = self.feature_tunk_push(rotate_colorpush, rotate_depthpush)
                    # Forward pass through branches
                    grasp = self.graspnet(interm_feat_grasp)
                    push = self.pushnet(interm_feat_push)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after_grasp = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), grasp.data.size(), align_corners=True)
                        flow_grid_after_push = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), push.data.size(), align_corners=True)

                    else:
                        flow_grid_after_grasp = F.affine_grid(Variable(affine_mat_after, requires_grad=False), grasp.data.size(), align_corners=True)
                        flow_grid_after_push = F.affine_grid(Variable(affine_mat_after, requires_grad=False), push.data.size(), align_corners=True)

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([F.interpolate(F.grid_sample(grasp, flow_grid_after_grasp, align_corners=True), scale_factor=16, mode='bilinear', align_corners=True),
                                       F.interpolate(F.grid_sample(push, flow_grid_after_push, align_corners=True), scale_factor=16, mode='bilinear', align_corners=True)])

            return output_prob, interm_feat_grasp, interm_feat_push

        else:
            self.output_prob = []

            # Apply rotations to intermediate features
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before_grasp = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), colorgrasp.size(), align_corners=True)
                flow_grid_before_push = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), colorpush.size(), align_corners=True)
            else:
                flow_grid_before_grasp = F.affine_grid(Variable(affine_mat_before, requires_grad=False), colorgrasp.size(), align_corners=True)
                flow_grid_before_push = F.affine_grid(Variable(affine_mat_before, requires_grad=False), colorpush.size(), align_corners=True)

            # Rotate images clockwise
            if self.use_cuda:
                rotate_colorgrasp = F.grid_sample(Variable(colorgrasp).cuda(), flow_grid_before_grasp, align_corners=True)
                rotate_depthgrasp = F.grid_sample(Variable(depthgrasp).cuda(), flow_grid_before_grasp, align_corners=True)
                rotate_colorpush = F.grid_sample(Variable(colorpush).cuda(), flow_grid_before_push, align_corners=True)
                rotate_depthpush = F.grid_sample(Variable(depthpush).cuda(), flow_grid_before_push, align_corners=True)
            else:
                rotate_colorgrasp = F.grid_sample(Variable(colorgrasp), flow_grid_before_grasp, align_corners=True)
                rotate_depthgrasp = F.grid_sample(Variable(depthgrasp), flow_grid_before_grasp, align_corners=True)
                rotate_colorpush = F.grid_sample(Variable(colorpush), flow_grid_before_push, align_corners=True)
                rotate_depthpush = F.grid_sample(Variable(depthpush), flow_grid_before_push, align_corners=True)

            # Compute intermediate features
            self.interm_feat_grasp = self.feature_tunk_grasp(rotate_colorgrasp, rotate_depthgrasp)
            self.interm_feat_push = self.feature_tunk_push(rotate_colorpush, rotate_depthpush)
            # Forward pass through branches
            grasp = self.graspnet(self.interm_feat_grasp)
            push = self.pushnet(self.interm_feat_push)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after_grasp = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), grasp.data.size(), align_corners=True)
                flow_grid_after_push = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), push.data.size(), align_corners=True)
            else:
                flow_grid_after_grasp = F.affine_grid(Variable(affine_mat_after, requires_grad=False), grasp.data.size(), align_corners=True)
                flow_grid_after_push = F.affine_grid(Variable(affine_mat_after, requires_grad=False), push.data.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([F.interpolate(F.grid_sample(grasp, flow_grid_after_grasp, align_corners=True), scale_factor=16, mode='bilinear', align_corners=True),
                                       F.interpolate(F.grid_sample(push, flow_grid_after_push, align_corners=True), scale_factor=16, mode='bilinear', align_corners=True)])

            return self.output_prob, self.interm_feat_grasp, self.interm_feat_push
