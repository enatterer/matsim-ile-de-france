#!/bin/bash

#SBATCH -J simulation_exp_dist_job

# Output and error files
#SBATCH --output=simulation_exp_dist_output_%j.log
#SBATCH --error=simulation_exp_dist_error_%j.log

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00

# Account and partition
#SBATCH --account=pn39mu
#SBATCH --partition=micro

# Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=elena.natterer@tum.de
#SBATCH --get-user-env

# Set up of execution environment
#SBATCH --export=none

#export HOME=/dss/dsshome1/06/ge49wav3
export FONTCONFIG_PATH=$HOME/test_java_font_4u/fonts
export FONTCONFIG_FILE=$HOME/test_java_font_4u/fonts/fonts.conf
export FC_FONT_PATH=$HOME/test_java_font_4u/fonts

# Load the openjdk module
module use /lrz/sys/share/modules/extfiles
module load openjdk/17.0.11_9

# Run the Java program with full arguments
java -cp ile_de_france/target/ile_de_france-1.5.0.jar org.eqasim.ile_de_france.RunSimulations1pctExpDistNotConnected > output_exp_dist.log 2>&1
