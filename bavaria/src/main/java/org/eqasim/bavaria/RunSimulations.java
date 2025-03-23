package org.eqasim.bavaria;

import java.nio.file.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;


public class RunSimulations extends SimulationRunnerBase {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations.class.getName());
    private static final int NUMBER_OF_SEEDS = 100;
    private static final int NUMBER_OF_THREADS = 6; // Adjust based on your system

    static public void main(String[] args) throws Exception {
        // Base directories
        String baseInputDir = "/Users/elenanatterer/Development/MATSim/eqasim-java-ile-de-france/bavaria/data/"; 
        String baseOutputDir = "/Users/elenanatterer/Development/MATSim/eqasim-java-ile-de-france/bavaria/output/base_cases/";

        // Get list of cities from input directory
        List<String> cities = getCityList(baseInputDir);
        LOGGER.info("Found cities: " + String.join(", ", cities));

        // Create a fixed thread pool
        ExecutorService executor = Executors.newFixedThreadPool(NUMBER_OF_THREADS);

        // Process each city
        for (String city : cities) {
            String cityInputDir = Paths.get(baseInputDir, city).toString();
            String cityOutputBaseDir = Paths.get(baseOutputDir, city).toString();
            System.out.println("City input directory: " + cityInputDir);
            System.out.println("City output base directory: " + cityOutputBaseDir);
            // Create city output directory if it doesn't exist
            try {
                Files.createDirectories(Paths.get(cityOutputBaseDir));
            } catch (IOException e) {
                LOGGER.severe("Could not create output directory for city " + city + ": " + e.getMessage());
                continue;
            }

            // Run simulations with different random seeds
            for (int seed = 1; seed <= NUMBER_OF_SEEDS; seed++) {
                final int finalSeed = seed;
                String seedOutputDir = Paths.get(cityOutputBaseDir, "seed_" + seed).toString();

                // Check if this simulation has already been completed
                if (isSimulationComplete(seedOutputDir)) {
                    LOGGER.info("Skipping completed simulation for " + city + " with seed " + finalSeed);
                    continue;
                }

                executor.submit(() -> {
                    try {
                        LOGGER.info("Starting simulation for " + city + " with seed " + finalSeed);
                        
                        // Prepare simulation parameters
                        String configFile = Paths.get(cityInputDir, city + "_config.xml").toString();
                        String networkFile = Paths.get(cityInputDir, city + "_network.xml.gz").toString();
                        String outputDirectory = Paths.get(cityOutputBaseDir, "seed_" + finalSeed).toString();
                        // Run the simulation
                        runSimulation(
                            city,
                            configFile,                    // Config file path
                            networkFile,                   // Network file (already in config)
                            null,              // working directory
                            outputDirectory,                 // output directory
                            finalSeed);
                        LOGGER.info("Completed simulation for " + city + " with seed " + finalSeed);
                    } catch (Exception e) {
                        LOGGER.log(Level.SEVERE, "Error in simulation for " + city + " with seed " + finalSeed, e);
                    }
                });
                break;
            }
            break;
        }

        // Shutdown the executor and wait for completion
        executor.shutdown();
        try {
            if (!executor.awaitTermination(300, TimeUnit.HOURS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException ie) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        LOGGER.info("All simulations completed");
    }

    private static List<String> getCityList(String baseDir) {
        List<String> cities = new ArrayList<>();
        try {
            Files.list(Paths.get(baseDir))
                .filter(Files::isDirectory)
                .map(path -> path.getFileName().toString())
                .forEach(cities::add);
        } catch (IOException e) {
            LOGGER.severe("Error reading city directories: " + e.getMessage());
        }
        return cities;
    }

    // private static boolean isSimulationComplete(String outputDir) {
    //     Path outputPath = Paths.get(outputDir);
    //     if (!Files.exists(outputPath)) {
    //         return false;
    //     }
        
    //     // Check for key output files that indicate a complete simulation
    //     Path outputEvents = outputPath.resolve("output_events.xml.gz");
    //     Path outputPlans = outputPath.resolve("output_plans.xml.gz");
        
    //     return Files.exists(outputEvents) && Files.exists(outputPlans);
    // }
}