package org.eqasim.bavaria;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Logger;
import java.nio.file.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

public abstract class SimulationRunnerBase {
    protected static final Logger LOGGER = Logger.getLogger(SimulationRunnerBase.class.getName());

    /**
     * Runs a MATSim simulation for a Bavarian city.
     *
     * @param configPath       Path to the city-specific config file
     * @param outputDirectory  Directory for simulation output
     * @param seed            Random seed for the simulation
     * @throws Exception if simulation fails
     */
    protected static void runSimulation(
            final String city,
            final String configPath,
            final String networkFile,
            final String workingDirectory,
            final String outputDirectory,
            final int seed
            ) throws Exception {

        // Ensure output directory exists
        createAndEmptyDirectory(outputDirectory);

        // Build the command
        List<String> arguments = new ArrayList<>(Arrays.asList(
            "java", "-Xmx32G",
            "-cp", "bavaria/target/bavaria-1.5.0.jar",
            "org.eqasim.bavaria.RunSimulations",
            "--config-path", configPath,
            "--output-path", outputDirectory,
            "--config:global.randomSeed", String.valueOf(seed),
            "--config:controler.outputDirectory", outputDirectory,
            "--config:global.numberOfThreads", "6",
            "--config:qsim.numberOfThreads", "6"
        ));

        LOGGER.info("Starting simulation with seed " + seed);
        LOGGER.info("Config: " + configPath);
        LOGGER.info("Output: " + outputDirectory);

        // Create and start the process
        Process process = new ProcessBuilder(arguments)
                .redirectOutput(new File(outputDirectory, "simulation_seed" + seed + ".log"))
                .redirectError(new File(outputDirectory, "simulation_seed" + seed + ".err.log"))
                .start();

        // Wait for process completion
        boolean interrupted = false;
        try {
            boolean finished = process.waitFor(72, TimeUnit.HOURS);  // 3 days timeout
            if (!finished) {
                process.destroy();
                throw new InterruptedException("Simulation timed out after 72 hours for seed " + seed);
            }
            int exitValue = process.exitValue();
            if (exitValue != 0) {
                throw new IOException("Simulation failed with exit code " + exitValue + " for seed " + seed);
            }
        } catch (InterruptedException e) {
            interrupted = true;
            process.destroy();
            throw e;
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        
        LOGGER.info("Completed simulation for seed " + seed);
    }

    /**
     * Creates or empties a directory for simulation output.
     */
    protected static void createAndEmptyDirectory(String directory) throws IOException {
        Path dirPath = Paths.get(directory);
        if (!Files.exists(dirPath)) {
            Files.createDirectories(dirPath);
            LOGGER.info("Created directory: " + directory);
        }
    }

    /**
     * Checks if a simulation has completed successfully.
     */
    protected static boolean isSimulationComplete(String directory) {
        Path dirPath = Paths.get(directory);
        
        // Check for essential output files
        String[] requiredFiles = {
            "output_events.xml.gz",
            "output_plans.xml.gz"
        };

        for (String file : requiredFiles) {
            Path filePath = dirPath.resolve(file);
            if (!Files.exists(filePath)) {
                return false;
            }
            // Also check if file is not empty
            try {
                if (Files.size(filePath) == 0) {
                    return false;
                }
            } catch (IOException e) {
                LOGGER.warning("Could not check file size for: " + filePath);
                return false;
            }
        }
        
        return true;
    }

    /**
     * Gets list of cities from the input directory.
     */
    protected static List<String> getCities(String baseInputDir) {
        List<String> cities = new ArrayList<>();
        try {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(baseInputDir))) {
                for (Path entry : stream) {
                    if (Files.isDirectory(entry)) {
                        cities.add(entry.getFileName().toString());
                    }
                }
            }
        } catch (IOException e) {
            LOGGER.severe("Error reading cities directory: " + e.getMessage());
        }
        return cities;
    }

    /**
     * Verifies that all required input files exist for a city.
     */
    protected static boolean verifyInputFiles(String cityInputDir, String cityName) {
        Path dirPath = Paths.get(cityInputDir);
        
        // Check for required input files
        String[] requiredFiles = {
            cityName + "_config.xml",
            cityName + "_network.xml.gz",
            // Add other required files here
        };

        for (String file : requiredFiles) {
            Path filePath = dirPath.resolve(file);
            if (!Files.exists(filePath)) {
                LOGGER.warning("Missing required file: " + filePath);
                return false;
            }
        }
        
        return true;
    }
}