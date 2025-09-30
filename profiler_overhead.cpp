#include "ExecutionProfiler.h"

#include <atomic>
#include <chrono>
#include <iostream>

using profiler::ExecutionProfiler;
using profiler::ScopedProfile;

namespace {

constexpr size_t kCounterId = 0;

inline void compilerBarrier() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

template <typename Fn>
double measureAverageNs(const Fn& fn, size_t iterations) {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        fn();
    }
    const auto end = clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(end - start);
    return elapsed.count() / static_cast<double>(iterations);
}

double measureLoopBaseline(size_t iterations) {
    return measureAverageNs([] {
        compilerBarrier();
    }, iterations);
}

double measureStartStop(size_t iterations) {
    ExecutionProfiler* profiler = ExecutionProfiler::instance();
    profiler->reset(kCounterId);
    return measureAverageNs([&] {
        profiler->startPoint(kCounterId);
        profiler->stopPoint(kCounterId);
    }, iterations);
}

double measureScopedProfile(size_t iterations) {
    ExecutionProfiler* profiler = ExecutionProfiler::instance();
    profiler->reset(kCounterId);
    return measureAverageNs([&] {
        ScopedProfile scope(kCounterId);
        compilerBarrier();
    }, iterations);
}

} // namespace

int main() {
    ExecutionProfiler* profiler = ExecutionProfiler::instance();
    profiler->resetAll();
    profiler->setName(kCounterId, "overhead probe");
    profiler->recalibrateCPUFreq();

    constexpr size_t kIterations = 100000;

    const double baselineNs = measureLoopBaseline(kIterations);
    const double startStopNs = measureStartStop(kIterations);
    const double scopedNs = measureScopedProfile(kIterations);

    std::cout << "Baseline loop: " << baselineNs << " ns" << '\n';
    std::cout << "startPoint/stopPoint: " << startStopNs << " ns" << '\n';
    std::cout << "ScopedProfile: " << scopedNs << " ns" << '\n';
    std::cout << "start/stop net overhead: " << (startStopNs - baselineNs) << " ns" << '\n';
    std::cout << "ScopedProfile net overhead: " << (scopedNs - baselineNs) << " ns" << '\n';

    return 0;
}
