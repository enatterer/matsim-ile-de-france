package org.eqasim.ile_de_france;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class RunSimulations1pmSingle extends SimulationRunnerBase {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pmSingle.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pm_simulations/pop_1pm_single_example/";
        String networkDirectory = "ile_de_france/data/pop_1pm_simulations/pop_1pm_single_example/networks/";

        // List all files in the directory
        Map<String, List<String>> networkFilesMap = getNetworkFiles(networkDirectory);

        // Create a fixed thread pool with 5 threads
        ExecutorService executor = Executors.newFixedThreadPool(1);

        LOGGER.info("Starting simulations");

        String folder = "networks_1000";
        List<String> networkFiles = networkFilesMap.get(folder);
        System.out.println("Network files: " + networkFiles);
        for (String networkFile : networkFiles) {
            final String finalNetworkFile = networkFile; // Final variable for lambda capture
            final String networkName = finalNetworkFile.replace(".xml.gz", "");
            System.out.println("Network name: " + networkName);

            final int randomSeed = 0;
            System.out.println("Random seed: " + randomSeed);
            final String outputDirectory = Paths.get(workingDirectory, "output_" + folder, networkName).toString();
            final String outputDirectorySeed = outputDirectory + "_seed_" + randomSeed;
                System.out.println("Submitting task for: " + networkName + " with seed: " + randomSeed);

                // Check if the file exists in the directory
                boolean fileExists = checkIfFileExists(outputDirectorySeed, "output_links.csv.gz");

                if (!outputDirectoryExists(outputDirectorySeed) || !fileExists) {
                    try {
                        createAndEmptyDirectory(outputDirectorySeed);
                        System.out.println("The directory " + outputDirectorySeed + " has been emptied.");
                    } catch (IOException e) {
                        System.err.println("An error occurred while creating or emptying the directory: " + e.getMessage());
                        continue; // Skip to the next iteration if directory creation or emptying fails
                    }

                    executor.submit(() -> {
                        System.out.println("Starting task for: " + finalNetworkFile);
                        try {
                            runSimulation(configPath, Paths.get("networks", folder, networkFile).toString(), outputDirectorySeed, workingDirectory, args, randomSeed, false, "1", "1", null);
                            // deleteUnwantedFiles(outputDirectorySeed);
                            // System.out.println("Deleted unwanted files for: " + networkFile);
                            System.out.println("Processed file: " + networkFile);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            LOGGER.log(Level.SEVERE, "Task interrupted for file: " + finalNetworkFile, e);
                        } 
                        catch (Exception e) {
                            LOGGER.log(Level.SEVERE, "Error processing file: " + networkFile, e);
                        }
                    });
                } else {
            LOGGER.info("Skipping simulation for existing output directory: " + outputDirectory);
        }

        // Shutdown the executor
        executor.shutdown();
        try {
            // Increase the wait time for all tasks to complete
            if (!executor.awaitTermination(300, TimeUnit.HOURS)) {
                executor.shutdownNow();
                if (!executor.awaitTermination(360, TimeUnit.SECONDS)) {
                    LOGGER.severe("Executor did not terminate");
                }
            }
        } catch (InterruptedException ie) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        LOGGER.info("Simulations completed");
    }
}
}