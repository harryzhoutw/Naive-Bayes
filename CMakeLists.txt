cmake_minimum_required(VERSION 3.14)
project(rfid-anomaly-detector)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Fetch spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)

include_directories(include)

add_executable(rfid_detector
    src/main.cpp
    src/service/RfidGaussianNaiveBayesService.cpp
)

target_link_libraries(rfid_detector PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog)