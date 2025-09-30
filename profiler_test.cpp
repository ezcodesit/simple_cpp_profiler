#include "ExecutionProfiler.h"

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

using profiler::ExecutionProfiler;
using profiler::ScopedProfile;

namespace {

constexpr size_t kLoopCounterId = 0;
constexpr size_t kManualCounterId = 1;
constexpr size_t kDisabledCounterId = 2;
constexpr size_t kMoveCounterId = 3;
constexpr size_t kParallelCounterId = 4;

constexpr auto kSleepDuration = std::chrono::milliseconds(2);

void busySleep() {
    std::this_thread::sleep_for(kSleepDuration);
}

bool testScopedLoop(ExecutionProfiler* profiler) {
    profiler->reset(kLoopCounterId);
    constexpr int kIterations = 5;
    for (int i = 0; i < kIterations; ++i) {
        ScopedProfile scope(kLoopCounterId);
        busySleep();
    }

    auto info = profiler->getInfo(kLoopCounterId);
    return info.counter_ == static_cast<uint64_t>(kIterations) && info.meanTimeNs_ > 0;
}

bool testManualStartStop(ExecutionProfiler* profiler) {
    profiler->reset(kManualCounterId);
    constexpr int kIterations = 3;
    for (int i = 0; i < kIterations; ++i) {
        profiler->startPoint(kManualCounterId);
        busySleep();
        profiler->stopPoint(kManualCounterId);
    }

    auto info = profiler->getInfo(kManualCounterId);
    return info.counter_ == static_cast<uint64_t>(kIterations) && info.deltaSumTicks_ > 0;
}

bool testDisabledScope(ExecutionProfiler* profiler) {
    profiler->reset(kDisabledCounterId);
    {
        ScopedProfile disabled(kDisabledCounterId, false);
        busySleep();
    }

    auto info = profiler->getInfo(kDisabledCounterId);
    return info.counter_ == 0 && info.deltaSumTicks_ == 0;
}

bool testScopedMove(ExecutionProfiler* profiler) {
    profiler->reset(kMoveCounterId);
    {
        ScopedProfile original(kMoveCounterId);
        ScopedProfile moved(std::move(original));
        (void)moved;
        busySleep();
    }

    auto info = profiler->getInfo(kMoveCounterId);
    return info.counter_ == 1 && info.meanTimeNs_ > 0;
}

bool testParallelAggregation(ExecutionProfiler* profiler) {
    profiler->reset(kParallelCounterId);

    std::thread workerA([&] {
        ScopedProfile scope(kParallelCounterId);
        busySleep();
    });

    std::thread workerB([&] {
        ScopedProfile scope(kParallelCounterId);
        busySleep();
    });

    workerA.join();
    workerB.join();

    auto info = profiler->getInfo(kParallelCounterId);
    return info.counter_ == 2 && info.meanTimeNs_ > 0;
}

struct TestCase {
    const char* name;
    bool (*fn)(ExecutionProfiler*);
};

} // namespace

int main() {
    ExecutionProfiler* profiler = ExecutionProfiler::instance();
    profiler->resetAll();
    profiler->setName(kLoopCounterId, "scoped loop");
    profiler->setName(kManualCounterId, "manual start/stop");
    profiler->setName(kDisabledCounterId, "disabled scope");
    profiler->setName(kMoveCounterId, "scoped move");
    profiler->setName(kParallelCounterId, "parallel scope");
    profiler->recalibrateCPUFreq();

    const std::array<TestCase, 5> tests = {{{"scoped loop", testScopedLoop},
                                            {"manual start/stop", testManualStartStop},
                                            {"disabled scope", testDisabledScope},
                                            {"scoped move", testScopedMove},
                                            {"parallel aggregation", testParallelAggregation}}};

    bool allPassed = true;
    for (const auto& test : tests) {
        const bool passed = test.fn(profiler);
        std::cout << (passed ? "PASS" : "FAIL") << ": " << test.name << '\n';
        allPassed = allPassed && passed;
    }

    return allPassed ? 0 : 1;
}
