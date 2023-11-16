#!/bin/bash

# run the Bonn dataset
./Examples/RGB-D/rgbd_tum ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/TUM1.yaml ./BONN/person_tracking2/ ./BONN/person_tracking2/associate.txt

# run by the Realsense D455 in real-world
./Examples/RGB-D/rgbd_realsense_D435i ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/Realsense_D455.yaml 20231101seq


# use the eva tools
python evaluate_ate_scale.py ../BONN/rgbd_bonn_crowd/groundtruth.txt ../BONN/rgbd_bonn_crowd/result_sigma1/CameraTrajectory.txt --plot plot.pdf --verbose

python evaluate_rpe.py ../TUM/walking_halfsphere/groundtruth.txt ../TUM/walking_halfsphere/result_sigma1/CameraTrajectory.txt --plot walkhalfrpe --fixed_delta --verbose


# use the evo tools
evo_ape tum groundtruth.txt result_sigma1/CameraTrajectory.txt -a -r full --plot

evo_traj tum traj1.txt –-plot

evo_ape tum groundtruth.txt result_sigma1/CameraTrajectory.txt -va -r full --plot
