#!/bin/bash

#SBATCH -J simulation_norm_dist_job

# Output and error files
#SBATCH --output=simulation_norm_dist_output_%j.log
#SBATCH --error=simulation_norm_dist_error_%j.log

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00

# Account and partition
#SBATCH --account=pn39mu
#SBATCH --partition=test

# Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=elena.natterer@tum.de
#SBATCH --get-user-env

# Set up of execution environment
export HOME=/dss/dsshome1/06/ge49wav3
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf
export FC_FONT_PATH=/usr/share/fonts

# Check accessible fonts
fc-list > accessible_fonts.log

# Run the Java program with full arguments
java -cp ile_de_france/target/ile_de_france-1.5.0.jar org.eqasim.ile_de_france.RunSimulations1pctNormDistNotConnected > output.log 2>&1
