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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class RunSimulations1pmMultipleThreadsManual {
    private static final Logger LOGGER = Logger.getLogger(RunSimulations1pmMultipleThreadsManual.class.getName());

    public static void main(String[] args) {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pm_with_policies/";
        String networkDirectory = "ile_de_france/data/pop_1pm_with_policies/networks/";

        // List all files in the directory
        List<String> xmlGzFiles = getNetworkFiles(networkDirectory);

        // Create and start threads for each network file, with a limit of 2 parallel threads
        List<Thread> threads = new ArrayList<>();
        for (String networkFile : xmlGzFiles) {
            Thread thread = new Thread(() -> {
                try {
                    String networkName = networkFile.replace(".xml.gz", "");
                    String outputDirectory = Paths.get(workingDirectory, "output/" + networkName).toString();
                    runSimulation(configPath, "networks/" + networkFile, outputDirectory, workingDirectory, args);
                    deleteUnwantedFiles(outputDirectory);
                    System.out.println("Processed and deleted file: " + networkFile);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            threads.add(thread);
        }

        // Run the threads in batches of 2
        int batchSize = 2;
        for (int i = 0; i < threads.size(); i += batchSize) {
            List<Thread> batch = threads.subList(i, Math.min(i + batchSize, threads.size()));
            batch.forEach(Thread::start);
            for (Thread thread : batch) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    e.printStackTrace();
                }
            }
        }
    }


    private static List<String> getNetworkFiles(String directoryPath) {
        File directory = new File(directoryPath);
        File[] filesList = directory.listFiles();
        List<String> xmlGzFiles = new ArrayList<>();
        if (filesList != null) {
            for (File file : filesList) {
                if (file.isFile() && file.getName().endsWith(".xml.gz")) {
                    xmlGzFiles.add(file.getName());
                }
            }
        } else {
            System.out.println("The specified directory does not exist or is not a directory.");
        }
        return xmlGzFiles;
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
        // Full path to the configuration file
        String fullConfigPath = Paths.get(workingDirectory, configPath).toString();

        // Configuration settings
        double flowCapacityFactor = 0.006;
        double storageCapacityFactor = 0.006;

        // Build command line parser
        CommandLine cmd = new CommandLine.Builder(args)
                .allowPrefixes("mode-choice-parameter", "cost-parameter")
                .build();

        // Initialize configurator and load config
        IDFConfigurator configurator = new IDFConfigurator();
        Config config = ConfigUtils.loadConfig(fullConfigPath, configurator.getConfigGroups());

        // Set additional configuration options
        config.controller().setOutputDirectory(outputDirectory);
        config.qsim().setFlowCapFactor(flowCapacityFactor);
        config.qsim().setStorageCapFactor(storageCapacityFactor);

        // Modify the network file parameter
        config.network().setInputFile(networkFile);

        // Add optional config groups and apply command line configuration
        configurator.addOptionalConfigGroups(config);
        cmd.applyConfiguration(config);

        // Create and configure scenario
        Scenario scenario = ScenarioUtils.createScenario(config);
        configurator.configureScenario(scenario);
        ScenarioUtils.loadScenario(scenario);
        configurator.adjustScenario(scenario);

        // Create and configure controller
        Controler controller = new Controler(scenario);
        configurator.configureController(controller);

        // Add necessary modules to the controller
        controller.addOverridingModule(new EqasimAnalysisModule());
        controller.addOverridingModule(new EqasimModeChoiceModule());
        controller.addOverridingModule(new IDFModeChoiceModule(cmd));

        // Run the simulation
        controller.run();
    }

    /**
     * Deletes all files and folders in the specified directory except for the specified files.
     *
     * @param outputDirectory The directory from which files and folders will be deleted.
     */
    private static void deleteUnwantedFiles(String outputDirectory) {
        Path dir = Paths.get(outputDirectory);
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path path : stream) {
                if (Files.isDirectory(path) ||
                        (!path.getFileName().toString().equals("output_links.csv.gz") &&
                                !path.getFileName().toString().equals("eqasim_pt.csv") &&
                                !path.getFileName().toString().equals("output_trips.csv.gz"))) {
                    Files.delete(path);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
