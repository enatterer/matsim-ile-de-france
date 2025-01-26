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

public class RunSimulations1pctSingle extends SimulationRunnerBase {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pctSingle.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pct_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pct_simulations/pop_1pct_basecase/";
        ExecutorService executor = Executors.newFixedThreadPool(1);
        final String networkName = "paris_1pct_network.xml.gz";
        for (int randomSeed = 45; randomSeed <= 100; randomSeed++) {
            final int finalRandomSeed = randomSeed;
            final String outputDirectorySeed = Paths.get(workingDirectory, "output_seed_" + finalRandomSeed).toString();
            boolean fileExists = checkIfFileExists(outputDirectorySeed, "output_links.csv.gz");
            if (!outputDirectoryExists(outputDirectorySeed) || !fileExists) {
                try {
                    createAndEmptyDirectory(outputDirectorySeed);
                    System.out.println("The directory " + outputDirectorySeed + " has been emptied.");
                } catch (IOException e) {
                    System.err.println("An error occurred while creating or emptying the directory: " + e.getMessage());
                }

                executor.submit(() -> {
                    System.out.println("Starting task for: " + networkName);
                    try {
                        LOGGER.info("Starting simulations");
                        runSimulation(configPath, networkName, outputDirectorySeed, workingDirectory, args, finalRandomSeed, true, "1", "1", null);
                        deleteUnwantedFiles(outputDirectorySeed);
                        System.out.println("Processed file: " + networkName);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        LOGGER.log(Level.SEVERE, "Task interrupted for file: " + networkName, e);
                    } 
                    catch (Exception e) {
                        LOGGER.log(Level.SEVERE, "Error processing file: " + networkName, e);
                    }
                });
            } else {
        LOGGER.info("Skipping simulation for existing output directory: " + outputDirectorySeed);
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