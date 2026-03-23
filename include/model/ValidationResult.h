#ifndef VALIDATION_RESULT_H
#define VALIDATION_RESULT_H

#include <string>

struct ValidationResult {
    bool isValid;
    double confidence;
    std::string reason;
};

#endif // VALIDATION_RESULT_H
