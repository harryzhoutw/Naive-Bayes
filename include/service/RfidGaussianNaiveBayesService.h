#ifndef RFID_GAUSSIAN_NAIVE_BAYES_SERVICE_H
#define RFID_GAUSSIAN_NAIVE_BAYES_SERVICE_H

#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "model/ValidationResult.h"

class RfidGaussianNaiveBayesService {
public:

    explicit RfidGaussianNaiveBayesService(const std::vector<std::string>& trainingData);
    ValidationResult validate(const std::string& rfid) const;

private:
    static constexpr size_t NUM_FEATURES = 6;
    static constexpr std::array<const char*, NUM_FEATURES> FEATURE_NAMES = {
        "Length", "Distinct Chars", "Entropy", "Numeric Value", "Letter Ratio", "Repeat Ratio"
    };

    struct GaussianParams {
        double mean;
        double std;
    };

    std::array<GaussianParams, NUM_FEATURES> featureParams_;
    double threshold_;

    std::array<double, NUM_FEATURES> extractFeatures(const std::string& rfid) const;
    double calculateLogLikelihood(const std::array<double, NUM_FEATURES>& features) const;
    static double logGaussianPdf(double x, double mean, double std);
    static double sigmoid(double x);
    static double calculateEntropy(const std::string& s);
    static double getNumericValue(const std::string& s);
    static std::string toUpperCase(const std::string& s);
    static std::string trim(const std::string& s);
};

#endif // RFID_GAUSSIAN_NAIVE_BAYES_SERVICE_H
