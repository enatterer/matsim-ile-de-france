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

public class RunSimulations1pmMultipleSeeds {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pmMultipleSeeds.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pm_basecase/";

        // Create a fixed thread pool with 2 threads
        ExecutorService executor = Executors.newFixedThreadPool(2);
        LOGGER.info("Starting simulations");

        for (int i = 1; i <= 10; i++) { // Run 10 iterations
            final String outputDirectory = Paths.get(workingDirectory, "output_seed_" + i).toString();
            final int finalI = i;
            executor.submit(() -> {
                try {
                    runSimulation(configPath,  outputDirectory, workingDirectory, args);
                    deleteUnwantedFiles(outputDirectory);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    LOGGER.log(Level.SEVERE, "Task interrupted for seed: %d".formatted(finalI), e);
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Error processing seed: %d".formatted(finalI), e);
                }
            });
        }

        // Shutdown the executor
        executor.shutdown();
        try {
            // Increase the wait time for all tasks to complete
            if (!executor.awaitTermination(24, TimeUnit.HOURS)) {
                executor.shutdownNow();
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    LOGGER.severe("Executor did not terminate");
                }
            }
        } catch (InterruptedException ie) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        LOGGER.info("Simulations completed");
    }

    /**
     * Runs the MATSim simulation with the given configuration path and output directory.
     *
     * @param configPath      The path to the configuration file.
     * @param outputDirectory The directory where output files will be stored.
     * @param workingDirectory The working directory.
     * @param args            Command line arguments.
     * @throws Exception if an error occurs during the simulation setup or execution.
     */
    public static void runSimulation(final String configPath, final String outputDirectory, final String workingDirectory, final String[] args) throws Exception {
        String fullConfigPath = Paths.get(workingDirectory, configPath).toString();
        final List<String> arguments = Arrays.asList("java", "-Xms32g", "-Xmx32g", "-cp",
                "ile_de_france/target/ile_de_france-1.5.0.jar",
                "org.eqasim.ile_de_france.RunSimulation1pm",
                "--config:global.numberOfThreads", "5",
                "--config:qsim.numberOfThreads", "5",
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
                throw new InterruptedException("Simulation process timed out: ");
            }
            int exitValue = process.exitValue();
            if (exitValue != 0) {
                throw new IOException("Simulation process failed with exit code " + exitValue);
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
        LOGGER.info("Completed simulation");
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
