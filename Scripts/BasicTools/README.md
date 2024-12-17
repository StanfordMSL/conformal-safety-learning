# BasicTools
Contains scripts for basic functionality like simulation experiment setup, visualization, etc.

## coordinate_utils
Euler angle functions

## covariance_utils
Inherited from https://github.com/chengine/splatnav/blob/master/ellipsoids/covariance_utils.py for Gaussian splat visualization

## dyn_system
General class for dynamical system and drone-specific system

## endpoint_sampler
General class for sampling task start-goal endpoints and specific cases

## experiment_info
General class for a simulated experiment setup and drone-specific cases (position obstacles and Gaussian splat navigation)
(Note: for backwards compatibility nerf is used instead of Gaussian splat terminology in code)

## geometric_helpers
Various geometry-related functions such as finding interior points, pruning constraints, and plotting balls

## helpers
Basic helper classes such as trajectory and rollout class and associated functions

## JE_compatibility
Wrapper class to convert from coordinates used in this code base to those used by Jun En's for hardware testing

## nerf_utils
Class for loading a Gaussian splat and rendering images, plotting in a server, walking in this 3D world etc.

## nn_utils
Class for MLP neural network and associated basic functions

## obs_sampler
General class for generating observations from a given state, with full-state, position-only, image-render cases 

## plotting_helpers
Provides basic functionalities for plotting drone trajectories and SUS region visualization

## safe_set
General class for determining whether a state is safe (with position, speed, cbf special cases)

## vision_helpers
Functions for working with image data, including human labeling of videos and point cloud visualization