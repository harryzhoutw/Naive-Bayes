#define _USE_MATH_DEFINES
#include <cmath>
#include "service/RfidGaussianNaiveBayesService.h"
#include <cctype>
#include <limits>
#include "spdlog/spdlog.h"

constexpr std::array<const char*, RfidGaussianNaiveBayesService::NUM_FEATURES> RfidGaussianNaiveBayesService::FEATURE_NAMES;

RfidGaussianNaiveBayesService::RfidGaussianNaiveBayesService(const std::vector<std::string>& trainingData) {
    // Collect unique normalized RFIDs
    std::unordered_set<std::string> uniqueSet;
    std::vector<std::string> uniqueList;

    for (const auto& rfid : trainingData) {
        std::string normalized = toUpperCase(trim(rfid));
        if (!normalized.empty() && uniqueSet.find(normalized) == uniqueSet.end()) {
            uniqueSet.insert(normalized);
            uniqueList.push_back(normalized);
        }
    }

    // Extract features from all training data
    std::vector<std::array<double, NUM_FEATURES>> allFeatures;
    for (const auto& rfid : uniqueList) {
        allFeatures.push_back(extractFeatures(rfid));
    }

    // Calculate Gaussian parameters (mean, std) for each feature
    spdlog::info("[Model Init] Learning Gaussian distribution from training data:");

    for (size_t f = 0; f < NUM_FEATURES; ++f) {
        double sum = 0.0, sumSq = 0.0;
        for (const auto& features : allFeatures) {
            sum += features[f];
            sumSq += features[f] * features[f];
        }

        double mean = sum / allFeatures.size();
        double variance = (sumSq / allFeatures.size()) - (mean * mean);
        double std = std::max(std::sqrt(variance), 0.1);  // Min std to avoid division by zero

        featureParams_[f] = {mean, std};
        spdlog::info("  {}: mean={}, std={}", FEATURE_NAMES[f], mean, std);
    }

    // Calculate threshold: minimum log-likelihood from training data - margin
    double minLogLikelihood = std::numeric_limits<double>::max();
    for (const auto& rfid : uniqueList) {
        double ll = calculateLogLikelihood(extractFeatures(rfid));
        minLogLikelihood = std::min(minLogLikelihood, ll);
    }

    threshold_ = minLogLikelihood - 1.0;
    spdlog::info("[Model Init] Min log-likelihood: {}, Threshold: {}", minLogLikelihood, threshold_);
}

ValidationResult RfidGaussianNaiveBayesService::validate(const std::string& rfid) const {
    std::string normalized = toUpperCase(trim(rfid));

    if (normalized.empty()) {
        return {false, 0.0, "Empty or null value"};
    }

    auto features = extractFeatures(normalized);
    double logLikelihood = calculateLogLikelihood(features);
    double confidence = sigmoid((logLikelihood - threshold_) / 5.0);

    std::string reason = fmt::format("log-likelihood={}", logLikelihood);

    if (logLikelihood < threshold_) {
        reason = fmt::format("{} < threshold={}", reason, threshold_);
        return {false, confidence, reason};
    }

    reason = fmt::format("{} >= threshold={}", reason, threshold_);
    return {true, confidence, reason};
}

std::array<double, RfidGaussianNaiveBayesService::NUM_FEATURES> RfidGaussianNaiveBayesService::extractFeatures(const std::string& rfid) const {
    if (rfid.empty()) {
        return {0, 0, 0, 0, 0, 0};
    }

    size_t len = rfid.length();

    // Distinct characters
    std::unordered_set<char> charSet(rfid.begin(), rfid.end());
    double distinctChars = static_cast<double>(charSet.size());

    // Letter count
    size_t letterCount = std::count_if(rfid.begin(), rfid.end(), ::isalpha);

    // Max repeat ratio
    std::unordered_map<char, int> freq;
    for (char c : rfid) {
        freq[c]++;
    }
    int maxRepeat = 0;
    for (const auto& p : freq) {
        maxRepeat = std::max(maxRepeat, p.second);
    }
    double repeatRatio = static_cast<double>(maxRepeat) / len;

    return {
        static_cast<double>(len),                    // Length
        distinctChars,                               // Distinct chars
        calculateEntropy(rfid),                      // Entropy
        getNumericValue(rfid),                       // Numeric value (log)
        static_cast<double>(letterCount) / len,      // Letter ratio
        repeatRatio                                  // Repeat ratio
    };
}

double RfidGaussianNaiveBayesService::calculateLogLikelihood(const std::array<double, NUM_FEATURES>& features) const {
    double logLL = 0.0;
    for (size_t f = 0; f < NUM_FEATURES; ++f) {
        logLL += logGaussianPdf(features[f], featureParams_[f].mean, featureParams_[f].std);
    }
    return logLL;
}

double RfidGaussianNaiveBayesService::logGaussianPdf(double x, double mean, double std) {
    double z = (x - mean) / std;
    return -0.5 * std::log(2 * M_PI) - std::log(std) - 0.5 * z * z;
}

double RfidGaussianNaiveBayesService::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double RfidGaussianNaiveBayesService::calculateEntropy(const std::string& s) {
    if (s.empty()) return 0.0;

    std::unordered_map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }

    double entropy = 0.0;
    double len = static_cast<double>(s.length());
    for (const auto& p : freq) {
        double prob = p.second / len;
        entropy -= prob * std::log2(prob);
    }
    return entropy;
}

double RfidGaussianNaiveBayesService::getNumericValue(const std::string& s) {
    if (s.length() > 15) {
        return std::log1p(std::numeric_limits<double>::max());
    }

    try {
        // Try to parse as HEX
        unsigned long long value = std::stoull(s, nullptr, 16);
        return std::log1p(static_cast<double>(value));
    } catch (...) {
        // Not valid HEX, calculate ASCII sum
        long sum = 0;
        for (char c : s) {
            sum += static_cast<int>(c);
        }
        return std::log1p(static_cast<double>(sum));
    }
}

std::string RfidGaussianNaiveBayesService::toUpperCase(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

std::string RfidGaussianNaiveBayesService::trim(const std::string& s) {
    auto start = std::find_if_not(s.begin(), s.end(), ::isspace);
    auto end = std::find_if_not(s.rbegin(), s.rend(), ::isspace).base();
    return (start < end) ? std::string(start, end) : "";
}
