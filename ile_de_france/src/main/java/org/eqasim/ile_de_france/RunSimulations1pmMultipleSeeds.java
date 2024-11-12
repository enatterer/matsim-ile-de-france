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

public class RunSimulations1pmMultipleSeeds extends SimulationRunnerBase{
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pmMultipleSeeds.class.getName());

    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pm_simulations/pop_1pm_cap_reduction_in_zone_1/";

        // Create a fixed thread pool with 2 threads
        ExecutorService executor = Executors.newFixedThreadPool(1);
        LOGGER.info("Starting simulations");

    }

    //     for (int i = 0; i <= 20; i++) { // Run 10 iterations
    //         final String outputDirectory = Paths.get(workingDirectory, "output_seed_" + i).toString();
    //         final int finalI = i;
    //         executor.submit(() -> {
    //             try {
    //                 runSimulation(configPath, Paths.get("networks", folder, networkFile).toString(), outputDirectory, workingDirectory, args, finalI, false, "4", "4", "32");
    //                 deleteUnwantedFiles(outputDirectory);
    //             } catch (InterruptedException e) {
    //                 Thread.currentThread().interrupt();
    //                 LOGGER.log(Level.SEVERE, "Task interrupted for seed: %d".formatted(finalI), e);
    //             } catch (Exception e) {
    //                 LOGGER.log(Level.SEVERE, "Error processing seed: %d".formatted(finalI), e);
    //             }
    //         });
    //     }

    //     // Shutdown the executor
    //     executor.shutdown();
    //     try {
    //         // Increase the wait time for all tasks to complete
    //         if (!executor.awaitTermination(24, TimeUnit.HOURS)) {
    //             executor.shutdownNow();
    //             if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
    //                 LOGGER.severe("Executor did not terminate");
    //             }
    //         }
    //     } catch (InterruptedException ie) {
    //         executor.shutdownNow();
    //         Thread.currentThread().interrupt();
    //     }
    //     LOGGER.info("Simulations completed");
    // }
}
