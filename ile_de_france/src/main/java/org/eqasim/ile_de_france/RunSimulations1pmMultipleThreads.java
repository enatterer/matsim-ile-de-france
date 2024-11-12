package org.eqasim.ile_de_france;

import org.eqasim.core.simulation.analysis.EqasimAnalysisModule;
import org.eqasim.core.simulation.mode_choice.EqasimModeChoiceModule;
import org.eqasim.ile_de_france.mode_choice.IDFModeChoiceModule;
import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.CommandLine;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.matsim.core.scenario.ScenarioUtils;

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

/*
 * Working parameters: 4 threads each, Xms10g, Xmx10g
 */

public class RunSimulations1pmMultipleThreads extends SimulationRunnerBase{
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pmMultipleThreads.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        
        String workingDirectory = "ile_de_france/data/pop_1pm_speed_reduction/";
        String networkDirectory = "ile_de_france/data/pop_1pm_speed_reduction/networks/";

        // List all files in the directory
        Map<String, List<String>> networkFilesMap = getNetworkFiles(networkDirectory);

        // Create a fixed thread pool with 15 threads
        ExecutorService executor = Executors.newFixedThreadPool(4);

        LOGGER.info("Starting simulations");

        for (int i = 1000; i <= 10000; i += 1000) {
            String folder = "networks_" + i;
            List<String> networkFiles = networkFilesMap.get(folder);
            if (networkFiles == null || networkFiles.isEmpty()) {
                continue;
            }

            for (String networkFile : networkFiles) {
                final String finalNetworkFile = networkFile; // Final variable for lambda capture
                final String networkName = finalNetworkFile.replace(".xml.gz", "");
                final String outputDirectory = Paths.get(workingDirectory, "output_" + folder, networkName).toString();
                LOGGER.info("Submitting task for: " + networkName);

                // Check if the file exists in the directory
                boolean fileExists = checkIfFileExists(outputDirectory, "output_links.csv.gz");

                if (!outputDirectoryExists(outputDirectory) || !fileExists) {
                    try {
                        createAndEmptyDirectory(outputDirectory);
                        LOGGER.info("The directory " + outputDirectory + " has been emptied.");
                    } catch (IOException e) {
                        LOGGER.severe("An error occurred while creating or emptying the directory: " + e.getMessage());
                        continue; // Skip to the next iteration if directory creation or emptying fails
                    }

                    executor.submit(() -> {
                        LOGGER.info("Starting task for: " + finalNetworkFile);
                        try {
                            runSimulation(configPath, Paths.get("networks", folder, networkFile).toString(), outputDirectory, workingDirectory, args, 0, false, "4", "4", "32");
                            deleteUnwantedFiles(outputDirectory);
                            LOGGER.info("Deleted unwanted files for: " + networkFile);
                            LOGGER.info("Processed file: " + networkFile);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            LOGGER.log(Level.SEVERE, "Task interrupted for file: " + finalNetworkFile, e);
                        } catch (Exception e) {
                            LOGGER.log(Level.SEVERE, "Error processing file: " + finalNetworkFile, e);
                        }
                    });
                } else {
                    LOGGER.info("Skipping simulation for existing output directory: " + outputDirectory);
                }
            }
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
