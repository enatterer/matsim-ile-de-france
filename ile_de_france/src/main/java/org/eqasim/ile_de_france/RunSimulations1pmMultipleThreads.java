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

public class RunSimulations1pmMultipleThreads {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pmMultipleThreads.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pm_with_policies/";
        String networkDirectory = "ile_de_france/data/pop_1pm_with_policies/networks/";

        // List all files in the directory
        Map<String, List<String>> networkFilesMap = getNetworkFiles(networkDirectory);

        // Create a fixed thread pool with 15 threads
        ExecutorService executor = Executors.newFixedThreadPool(6);

        LOGGER.info("Starting simulations");

        for (int i = 1500; i <= 5000; i += 100) {
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
                            runSimulation(configPath, Paths.get("networks", folder, networkFile).toString(), outputDirectory, workingDirectory, args);
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

    public static void createAndEmptyDirectory(String directory) throws IOException {
        Path dirPath = Paths.get(directory);
        LOGGER.info("Creating or emptying directory: " + directory);

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

    private static void deleteRecursively(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
                for (Path entry : stream) {
                    deleteRecursively(entry);
                }
            }
        }
        Files.delete(path);
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
    public static void runSimulation(final String configPath, final String networkFile, final String outputDirectory, final String workingDirectory, final String[] args) throws Exception {
        String fullConfigPath = Paths.get(workingDirectory, configPath).toString();
        LOGGER.info("Running simulation with config: " + fullConfigPath + ", network file: " + networkFile + ", output directory: " + outputDirectory);

        final List<String> arguments = Arrays.asList("java", "-Xms32g", "-Xmx32g","-cp",
                "ile_de_france/target/ile_de_france-1.5.0.jar",
                "org.eqasim.ile_de_france.RunSimulation1pm",
                "--config:global.numberOfThreads", "12",
                "--config:qsim.numberOfThreads", "12",
                "--config:network.inputNetworkFile", networkFile,
                "--config:controler.outputDirectory", outputDirectory,
                "--config-path", fullConfigPath);

        Process process = new ProcessBuilder(arguments)
                .redirectOutput(new File(outputDirectory + ".log"))
                .redirectError(new File(outputDirectory + ".error.log"))
                .start();
        LOGGER.info("Started process: " + outputDirectory);

        boolean interrupted = false;
        try {
            boolean finished = process.waitFor(10, TimeUnit.HOURS);  // Increase wait time
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

    /**
     * Deletes all files and folders in the specified directory except for the specified files.
     *
     * @param outputDirectory The directory from which files and folders will be deleted.
     */
    private static void deleteUnwantedFiles(String outputDirectory) {
        Path dir = Paths.get(outputDirectory);
        if (!Files.exists(dir)) {
            LOGGER.warning("Output directory does not exist: " + outputDirectory);
            return;
        }

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path path : stream) {
                if (Files.isDirectory(path)) {
                    LOGGER.info("Deleting directory: " + path);
                    deleteDirectoryRecursively(path);
                } else {
                    String fileName = path.getFileName().toString();
                    if (!fileName.equals("output_links.csv.gz")
                            && !fileName.equals("eqasim_pt.csv")
                            && !fileName.equals("output_trips.csv.gz")) {
                        Files.delete(path);
                        LOGGER.info("Deleted file: " + path);
                    } else {
                        LOGGER.info("Skipping file: " + path);
                    }
                }
            }
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error deleting files in directory: " + outputDirectory, e);
        }
    }

    /**
     * Recursively deletes a directory and its contents.
     *
     * @param directory The directory to be deleted.
     * @throws IOException If an I/O error occurs.
     */
    private static void deleteDirectoryRecursively(Path directory) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    deleteDirectoryRecursively(entry);
                } else {
                    Files.delete(entry);
                    LOGGER.info("Deleted file: " + entry);
                }
            }
        }
        Files.delete(directory);
        LOGGER.info("Deleted directory: " + directory);
    }
}
