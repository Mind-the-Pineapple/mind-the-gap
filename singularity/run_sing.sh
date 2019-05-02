#!/bin/bash

datapath=/data/project/mindgap/data
singularity run -c \
                -B ~/Code/mind-the-gap:\code \
                -B $datapath:\code/data \
                $(dirname$0)/mind.img
