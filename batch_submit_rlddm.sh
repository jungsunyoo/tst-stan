#!/bin/bash

for states in 2 3 4 5; do
    for subject in $(cat state${states}_subjects.txt); do
        sbatch rlddm-fit-slurm.sub $subject $states
    done
done
