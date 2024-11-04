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

# Set up of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env

# Load modules
module load openjdk

# Navigate to the directory containing the Java program
# cd /path/to/your/java/program

# Execute the Java program
java -cp ile_de_france/target/ile_de_france-1.5.0.jar org.eqasim.ile_de_france.RunSimulations1pctNormDistNotConnected
