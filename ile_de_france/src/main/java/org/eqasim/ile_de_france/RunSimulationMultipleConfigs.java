package org.eqasim.ile_de_france;

import org.eqasim.core.simulation.analysis.EqasimAnalysisModule;
import org.eqasim.core.simulation.mode_choice.EqasimModeChoiceModule;
import org.eqasim.ile_de_france.mode_choice.IDFModeChoiceModule;
import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.CommandLine;
import org.matsim.core.config.CommandLine.ConfigurationException;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.matsim.core.scenario.ScenarioUtils;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * TODO: Absolute paths are used, change them to relative paths.
 */

public class RunSimulationMultipleConfigs {
    static public void main(String[] args) throws Exception {
        // Configuration settings
        String configPath = "paris_1pm_config.xml";
        String workingDirectory = "/Users/elenanatterer/Development/MATSim/eqasim-java/ile_de_france/data/pop_1pm/pop_1pm_with_policies/";

        // Generate the list of network files dynamically
        List<String> networkFiles = generateNetworkFiles();

        // Loop over all network files and run simulations
        for (String networkFile : networkFiles) {
            // Extract the network name to create the corresponding output directory
            String networkName = networkFile.replace(".xml.gz", "");
            String outputDirectory = Paths.get(workingDirectory,"output_policies/output_" + networkName).toString();

            // Call the method to run the simulation
            runSimulation(configPath, networkFile, outputDirectory, workingDirectory, args);
        }
    }

    /**
     * Runs the MATSim simulation with the given configuration path and output directory.
     *
     * @param configPath      The path to the configuration file.
     * @param outputDirectory The directory where output files will be stored.
     * @param args            Command line arguments.
     * @throws Exception if an error occurs during the simulation setup or execution.
     */
    public static void runSimulation(final String configPath, final String networkFile, final String outputDirectory, final String workingDirectory, final String[] args) throws Exception {
        // Set the working directory
        System.setProperty("user.dir", workingDirectory);

        // Full path to the configuration file
        String fullConfigPath = Paths.get(workingDirectory, configPath).toString();

        // Configuration settings
        double flowCapacityFactor = 1e9;
        double storageCapacityFactor = 1e9;

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
     * Generates a list of network file names dynamically - we are generating network files for Arrondissement 1, (1, 2), (1, 2, 3), etc. In total, 57 files are generated.
     *
     * @return A list of network file names.
     */
    private static List<String> generateNetworkFiles() {
        List<String> networkFiles = new ArrayList<>();
        for (int i = 1; i <= 20; i++) {
            networkFiles.add(String.format("networks/network_district_%d.xml.gz", i));
            if (i < 20) {
                networkFiles.add(String.format("networks/network_district_%d_%d.xml.gz", i, i + 1));
                if (i < 19) {
                    networkFiles.add(String.format("networks/network_district_%d_%d_%d.xml.gz", i, i + 1, i + 2));
                }
            }
        }
        return networkFiles;
    }
}

