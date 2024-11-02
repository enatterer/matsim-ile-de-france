# !/bin/bash

SBATCH --job-name=simulation_norm_dist_job
SBATCH --output=simulation_norm_dist_output_%j.log
SBATCH --error=simulation_norm_dist_error_%j.log
SBATCH --ntasks=4
SBATCH --cpus-per-task=12
SBATCH --time=00:30:00
SBATCH --mem=40G
SBATCH --partition=test
SBATCH --account=test

module load openjdk

# Navigate to the directory containing the Java program
# cd /path/to/your/java/program

# Execute the Java program
java -cp ile_de_france/target/ile_de_france-1.5.0.jar org.eqasim.ile_de_france.RunSimulations1pctNormDistNotConnected
