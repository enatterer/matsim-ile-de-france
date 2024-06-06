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
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class RunSimulations1pctMultipleThreads {
    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pct_config.xml";
        String workingDirectory = "ile_de_france/data/pop_1pct_with_policies/";
        String networkDirectory = "ile_de_france/data/pop_1pct_with_policies/networks/";

        // List all files in the directory
        List<String> xmlGzFiles = getNetworkFiles(networkDirectory);

        // Create a fixed thread pool with 10 threads
        ExecutorService executor = Executors.newFixedThreadPool(10);

        // Loop over all network files and submit simulations to the executor
        for (String networkFile : xmlGzFiles) {
            executor.submit(() -> {
                try {
                    String networkName = networkFile.replace(".xml.gz", "");
                    String outputDirectory = Paths.get(workingDirectory, "output/" + networkName).toString();
                    runSimulation(configPath, "networks/" + networkFile, outputDirectory, workingDirectory, args);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }

        // Shutdown the executor
        executor.shutdown();
        // Wait for all tasks to complete
        if (!executor.awaitTermination(60, TimeUnit.MINUTES)) {
            executor.shutdownNow();
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
        double flowCapacityFactor = 0.06;
        double storageCapacityFactor = 0.06;

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
}
