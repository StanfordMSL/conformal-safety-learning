# Policies
Contains scripts for control of drone

## obs_avoid_constr
General class obstacle avoidance constraints that can be used in MPC. Two special cases for ellipsoids and polytopes.

## point_cloud_avoid
Class for avoiding nearby point cloud using Gaussian splat and one-sample NNCP.

## policy
General class for what a policy should feature to interface with the experiment class i.e., to be run in our simulation setup

## scp_mpc
Main class for generating MPC policy for drone