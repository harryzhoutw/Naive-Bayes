#include "service/RfidGaussianNaiveBayesService.h"
#include "model/ValidationResult.h" // Include ValidationResult explicitly
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string> // For std::string
#include <cstdlib> // For exit()
#include "spdlog/spdlog.h"

using json = nlohmann::json;

// Define the default path for the test data configuration file
const std::string DEFAULT_TEST_PATH = "test/test_data.json";

// --- Helper Functions ---

// Function to load and parse configuration from a JSON file
json loadConfig(const std::string& configPath) {
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) {
        spdlog::error("Error: Cannot open config file: {}", configPath);
        exit(1); // Exit if config file cannot be opened
    }

    json config;
    try {
        configFile >> config;
    } catch (const json::parse_error& e) {
        spdlog::error("Error: Failed to parse config file '{}': {}", configPath, e.what());
        exit(1); // Exit if parsing fails
    }
    return config;
}

// Function to extract data vectors from the JSON configuration
void extractData(const json& config,
                 std::vector<std::string>& normalRfids,
                 std::vector<std::string>& testNormal,
                 std::vector<std::string>& testAnomaly) {
    try {
        // Using .at() for robust error handling if keys are missing
        normalRfids = config.at("normalRfids").get<std::vector<std::string>>();
        testNormal = config.at("testNormal").get<std::vector<std::string>>();
        testAnomaly = config.at("testAnomaly").get<std::vector<std::string>>();
    } catch (const json::out_of_range& e) {
        spdlog::error("Error: Missing key in config file: {}", e.what());
        exit(1); // Exit if required keys are missing
    } catch (const json::type_error& e) {
        spdlog::error("Error: Incorrect type for key in config file: {}", e.what());
        exit(1); // Exit if data types are incorrect
    }
}

// Function to run tests for normal RFIDs
void runNormalTests(const RfidGaussianNaiveBayesService& service, const std::vector<std::string>& testNormal) {
    spdlog::info("=== Testing Normal RFIDs ===");
    for (const auto& rfid : testNormal) {
        ValidationResult result = service.validate(rfid);
        spdlog::info("RFID: {} -> {} (confidence: {})", rfid, (result.isValid ? "VALID" : "INVALID"), result.confidence);
    }
}

// Function to run tests for anomaly RFIDs and return detection count
int runAnomalyTests(const RfidGaussianNaiveBayesService& service, const std::vector<std::string>& testAnomaly) {
    spdlog::info("=== Testing Anomaly RFIDs ===");
    int detected = 0;
    for (const auto& rfid : testAnomaly) {
        ValidationResult result = service.validate(rfid);
        spdlog::info("RFID: {} -> {} ({})", rfid, (result.isValid ? "VALID" : "INVALID"), result.reason);
        if (!result.isValid) {
            detected++;
        }
    }
    return detected;
}

// Function to print the summary of anomaly detection
void printSummary(int detected, size_t totalAnomaly) {
    spdlog::info("=== Summary ===");
    if (totalAnomaly > 0) {
        spdlog::info("Anomaly detection rate: {}/{} ({}%)", detected, totalAnomaly, (100.0 * detected / totalAnomaly));
    } else {
        spdlog::info("No anomaly tests were run.");
    }
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    std::string configPath = DEFAULT_TEST_PATH; // Use the global constant
    if (argc > 1) {
        configPath = argv[1];
    }

    spdlog::info("=== RFID Anomaly Detector ===");
    spdlog::info("Loaded config from: {}", configPath);

    json config = loadConfig(configPath);

    std::vector<std::string> normalRfids;
    std::vector<std::string> testNormal;
    std::vector<std::string> testAnomaly;
    extractData(config, normalRfids, testNormal, testAnomaly);
    
    // Model Initialization
    RfidGaussianNaiveBayesService rfidGaussianNaiveBayesService(normalRfids);

    runNormalTests(rfidGaussianNaiveBayesService, testNormal);
    
    int detectedAnomalies = runAnomalyTests(rfidGaussianNaiveBayesService, testAnomaly);
    
    printSummary(detectedAnomalies, testAnomaly.size());

    return 0;
}
