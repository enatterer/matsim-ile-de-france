package org.eqasim.ile_de_france;

import java.io.IOException;
import java.nio.file.*;
import java.util.concurrent.*;
import java.util.logging.*;

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

public abstract class SimulationRunnerBase {
    protected static final Logger LOGGER = Logger.getLogger(SimulationRunnerBase.class.getName());

    protected static void deleteUnwantedFiles(String outputDirectory) throws IOException {
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
                            && !fileName.equals("eqasim_trips.csv")) {
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

    protected static void deleteDirectoryRecursively(Path directory) throws IOException {
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

    protected static void deleteRecursively(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
                for (Path entry : stream) {
                    deleteRecursively(entry);
                }
            }
        }
        Files.delete(path);
    }
}

