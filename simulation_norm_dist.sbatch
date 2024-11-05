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
# module load openjdk

# Navigate to the directory containing the Java program
# cd /path/to/your/java/program

# Execute the Java program
# java -cp "ile_de_france/target/ile_de_france-1.5.0.jar" org.eqasim.ile_de_france.RunSimulations1pctNormDistNotConnected

# Run the Java program with full arguments
java -Djava.awt.headless=true -cp ile_de_france/target/ile_de_france-1.5.0.jar \
    org.eqasim.ile_de_france.RunSimulations1pctNormDistNotConnected \
    --config:global.numberOfThreads 12 \
    --config:qsim.numberOfThreads 12 \
    --config:network.inputNetworkFile networks/networks_1000/network_d_2_4_6_7_12.xml.gz \
    --config:controler.outputDirectory ile_de_france/data/pop_1pct_simulations/pop_1pct_cap_reduction/norm_dist_not_connected_5k/output_networks_5000/network_d_2_4_6_7_12 \
    --config-path ile_de_france/data/pop_1pct_simulations/pop_1pct_cap_reduction/norm_dist_not_connected_5k/paris_1pct_config.xml

