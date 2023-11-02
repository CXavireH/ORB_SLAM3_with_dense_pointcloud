#!/bin/bash

# run the Bonn dataset
./Examples/RGB-D/rgbd_tum ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/TUM1.yaml ./BONN/person_tracking2/ ./BONN/person_tracking2/associate.txt

# run by the Realsense D455 in real-world
./Examples/RGB-D/rgbd_realsense_D435i ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/Realsense_D455.yaml 20231101seq
