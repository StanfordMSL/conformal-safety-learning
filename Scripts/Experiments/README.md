# Experiments
Contains scripts for generating the experimental results and figures

## backup_demo
Generates example results for position obstacles using the modified policy with backup safety mode

## backup_demo_nerf
Generates example simulation results for Gaussian splat (nerf) navigation and saves policy for hardware testing

## backup_systematic
Generates systematic results for comparing the backup safety mode to naive baseline for policy modification

## compare_backup_nerf
A diagnostic to generate simulation results for the original and backup policy we expect to see in hardware tests

## hardware_comparison
Script for processing and generating results for the hardware tests

## nerf_label
Script for human labeling of navigation in nerf

## no_track_demo
Analogue to backup_demo except for the naive baseline which directly constrains to avoid SUS region

## two_sample_plots
Generates the illustrative 2D examples for the NNCP theory section

## visuomotor_demo
Generates videos and images of p-value runtime monitor for visuomotor policy

## visuomotor_label
Script for human labeling of visuomotor policy

## visuomotor_recon_demo
Generates example image reconstructions using PCA and autoencoder models

## visuomotor_systematic
Generate systematic representation learning comparison across train-test splits

## warning_demo
Perform example fit and run of the warning system with position obstacles

## warning_systematic
Systematic comparison of NNCP against ablations and ML