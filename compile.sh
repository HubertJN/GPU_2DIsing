#!/bin/bash

gcc create_cluster_set.c functions/read_input_variables.c functions/comparison.c -o bin/create_cluster_set -Wall -Wextra
gcc calculate_cluster.c functions/read_input_variables.c functions/find_cluster_size.c functions/read_input_grid.c -o bin/calculate_cluster -Wall
