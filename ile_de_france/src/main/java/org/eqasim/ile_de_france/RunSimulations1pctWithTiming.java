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
import java.awt.GraphicsEnvironment;
import java.time.Duration;
import java.time.Instant;

public class RunSimulations1pctExpDistNotConnected extends SimulationRunnerBase {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pctExpDistNotConnected.class.getName());

    static public void main(String[] args) throws Exception {
        // Start timing the overall process
        Instant startTime = Instant.now();
        LOGGER.info("Starting simulations at: " + startTime);

        // Configuration settings
        String configPath = "paris_1pct_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pct_cap_reduction/exp_dist_not_connected_5k/";
        String networkDirectory = "ile_de_france/data/pop_1pct_cap_reduction/exp_dist_not_connected_5k/networks/";

        // List all files in the directory
        Map<String, List<String>> networkFilesMap = getNetworkFiles(networkDirectory);
        
        // Debug print for network directory
        System.out.println("Network directory: " + networkDirectory);
        System.out.println("Working directory: " + workingDirectory);
        
        // Print network files map
        System.out.println("\nNetwork files found:");
        for (Map.Entry<String, List<String>> entry : networkFilesMap.entrySet()) {
            System.out.println("\nFolder: " + entry.getKey());
            for (String file : entry.getValue()) {
                System.out.println("  - " + file);
            }
        }

        // Create a fixed thread pool with 5 threads
        ExecutorService executor = Executors.newFixedThreadPool(6);

        for (int i = 1000; i <= 5000; i += 1000) {
            String folder = "networks_" + i;
            List<String> networkFiles = networkFilesMap.get(folder);
            if (networkFiles == null || networkFiles.isEmpty()) {
                System.out.println("No network files found for folder: " + folder);
                continue;
            }
            for (String networkFile : networkFiles) {
                final String finalNetworkFile = networkFile; // Final variable for lambda capture
                final String networkName = finalNetworkFile.replace(".xml.gz", "");
                System.out.println("\nProcessing network: " + networkName);
                final String outputDirectory = Paths.get(workingDirectory, "output_" + folder, networkName).toString();
                System.out.println("Output directory: " + outputDirectory);
                System.out.println("Network file path: " + Paths.get("networks", folder, networkFile).toString());
                boolean fileExists = checkIfFileExists(outputDirectory, "output_links.csv.gz");

                if (!outputDirectoryExists(outputDirectory) || !fileExists) {
                    try {
                        createAndEmptyDirectory(outputDirectory);
                        System.out.println("The directory " + outputDirectory + " has been emptied.");
                    } catch (IOException e) {
                        System.err.println("An error occurred while creating or emptying the directory: " + e.getMessage());
                        continue; // Skip to the next iteration if directory creation or emptying fails
                    }

                    executor.submit(() -> {
                        Instant simStartTime = Instant.now();
                        System.out.println("Starting task for: " + finalNetworkFile + " at: " + simStartTime);
                        try {
                            runSimulation(configPath, Paths.get("networks", folder, networkFile).toString(), outputDirectory, workingDirectory, args, 0, true, "8", "8", null);
                            deleteUnwantedFiles(outputDirectory);
                            Instant simEndTime = Instant.now();
                            Duration simDuration = Duration.between(simStartTime, simEndTime);
                            System.out.println("Simulation completed for: " + finalNetworkFile);
                            System.out.println("Duration: " + formatDuration(simDuration));
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
        
        // Calculate and log total duration
        Instant endTime = Instant.now();
        Duration totalDuration = Duration.between(startTime, endTime);
        LOGGER.info("Simulations completed at: " + endTime);
        LOGGER.info("Total duration: " + formatDuration(totalDuration));
    }

    private static String formatDuration(Duration duration) {
        long hours = duration.toHours();
        long minutes = duration.toMinutesPart();
        long seconds = duration.toSecondsPart();
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }
}
