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

public class RunSimulations1pctMultipleThreadsSingleDistricts extends SimulationRunnerBase {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pctMultipleThreadsSingleDistricts.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pct_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pct_simulations/pop_1pct_cap_reduction/";
        String networkDirectory = "ile_de_france/data/pop_1pct_simulations/pop_1pct_cap_reduction/networks/";

        // List all files in the directory
        Map<String, List<String>> networkFilesMap = getNetworkFiles(networkDirectory);

        // Create a fixed thread pool with 5 threads
        ExecutorService executor = Executors.newFixedThreadPool(2);

        for (int i = 1000; i <= 16000; i += 1000) {
            String folder = "networks_" + i;
            List<String> networkFiles = networkFilesMap.get(folder);
            if (networkFiles == null || networkFiles.isEmpty()) {
                continue;
            }

            for (String networkFile : networkFiles) {
                for (int seed = 1; seed <= 20; seed++) {
                    final String finalNetworkFile = networkFile; // Final variable for lambda capture
                    final String networkName = finalNetworkFile.replace(".xml.gz", "");
                    final String outputDirectory_start = Paths.get(workingDirectory, "output_" + folder, networkName).toString();
                    final String outputDirectory = outputDirectory_start + "_seed_" + seed;
                    System.out.println("Submitting task for: " + networkName);
                    final int finalSeed = seed;

                    // Check if the file exists in the directory
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
                            System.out.println("Starting task for: " + finalNetworkFile);
                            try {
                                runSimulation(configPath, Paths.get("networks", folder, networkFile).toString(), outputDirectory, workingDirectory, finalSeed, args);
                                deleteUnwantedFiles(outputDirectory);
                                System.out.println("Deleted unwanted files for: " + networkFile);
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

    public static void createAndEmptyDirectory(String directory) throws IOException {
        Path dirPath = Paths.get(directory);

        if (!Files.exists(dirPath)) {
            Files.createDirectories(dirPath);
        } else if (Files.isDirectory(dirPath)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath)) {
                for (Path entry : stream) {
                    deleteRecursively(entry);
                }
            }
        } else {
            throw new IOException("The path specified is not a directory: " + directory);
        }
    }

    public static boolean checkIfFileExists(String directory, String fileName) {
        Path dirPath = Paths.get(directory);
        Path filePath = dirPath.resolve(fileName);
        boolean exists = Files.exists(filePath) && !Files.isDirectory(filePath);
        LOGGER.info("Checking if file exists: " + filePath + " - " + exists);
        return exists;
    }

    private static Map<String, List<String>> getNetworkFiles(String directoryPath) {
        File mainDirectory = new File(directoryPath);
        File[] subDirs = mainDirectory.listFiles(File::isDirectory);

        if (subDirs == null) {
            System.out.println("The specified directory does not exist or is not a directory.");
            return Map.of();
        }

        return Arrays.stream(subDirs)
                .collect(Collectors.toMap(
                        File::getName,
                        subDir -> {
                            File[] filesList = subDir.listFiles((dir, name) -> name.endsWith(".xml.gz"));
                            List<String> xmlGzFiles = new ArrayList<>();
                            if (filesList != null) {
                                for (File file : filesList) {
                                    if (file.isFile()) {
                                        xmlGzFiles.add(file.getName());
                                    }
                                }
                                // Sort the list of file names
                                Collections.sort(xmlGzFiles);
                            }
                            return xmlGzFiles;
                        }
                ));
    }

    private static boolean outputDirectoryExists(String outputDirectory) {
        File dir = new File(outputDirectory);
        boolean exists = dir.exists() && dir.isDirectory();
        LOGGER.info("Checking if output directory exists: " + outputDirectory + " - " + exists);
        return exists;
    }

    /**
     * Runs the MATSim simulation with the given configuration path and output directory.
     *
     * @param configPath      The path to the configuration file.
     * @param networkFile     The network file to use for the simulation.
     * @param outputDirectory The directory where output files will be stored.
     * @param workingDirectory The working directory.
     * @param args            Command line arguments.
     * @throws Exception if an error occurs during the simulation setup or execution.
     */
    public static void runSimulation(final String configPath, final String networkFile, final String outputDirectory, final String workingDirectory, final int seed, final String[] args) throws Exception {
        // Full path to the configuration file
        String fullConfigPath = Paths.get(workingDirectory, configPath).toString();

        final List<String> arguments = Arrays.asList("java", "-Xms40g", "-Xmx40g", "-cp",
                "ile_de_france/target/ile_de_france-1.5.0.jar",
                "org.eqasim.ile_de_france.RunSimulation1pct",
                "--config:global.numberOfThreads", "12",
                "--config:qsim.numberOfThreads", "12",
                "--config:global.randomSeed", String.valueOf(seed),
                "--config:network.inputNetworkFile", networkFile,
                "--config:controler.outputDirectory", outputDirectory,
                "--config-path", fullConfigPath);
        // arguments.forEach(System.out::println);

        Process process = new ProcessBuilder(arguments)
                .redirectOutput(new File(outputDirectory + ".log"))
                .redirectError(new File(outputDirectory + ".error.log"))
                .start();
        System.out.println("Started process: " + outputDirectory);

        boolean interrupted = false;
        try {
            boolean finished = process.waitFor(3000, TimeUnit.HOURS);  // Increase wait time
            if (!finished) {
                process.destroy();  // destroy process if it times out
                throw new InterruptedException("Simulation process timed out: " + networkFile);
            }
            int exitValue = process.exitValue();
            if (exitValue != 0) {
                throw new IOException("Simulation process failed with exit code " + exitValue + ": " + networkFile);
            }
        } catch (InterruptedException e) {
            interrupted = true;
            process.destroy();  // ensure process is destroyed if interrupted
            throw e;  // rethrow the exception to be handled in the calling method
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        LOGGER.info("Completed simulation for: " + networkFile);
    }
}
