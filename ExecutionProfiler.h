/// To use just include this file.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <thread>
#include <functional>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif


// ExecutionProfiler - main profiler class
// can be used for several threads

namespace profiler {

namespace detail {

/// Utility responsible for retrieving serialized timestamp samples.
/// On x86 we rely on rdtsc/rdtscp. On other platforms the timestamps
/// are based on the monotonic steady clock in nanoseconds.
class TimestampReader {
public:
    static inline uint64_t readStart() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_lfence();
        return __rdtsc();
#else
        return monotonicNow();
#endif
    }

    static inline uint64_t readStop() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        unsigned aux;
        auto value = __rdtscp(&aux);
        _mm_lfence();
        return value;
#else
        return monotonicNow();
#endif
    }

private:
    static inline uint64_t monotonicNow() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    }
};

/// Platform-specific helpers that do not belong to the public API.
inline int64_t queryThreadId() {
#if defined(__linux__)
    return static_cast<int64_t>(syscall(SYS_gettid));
#elif defined(__APPLE__)
    uint64_t threadId = 0;
    pthread_threadid_np(nullptr, &threadId);
    return static_cast<int64_t>(threadId);
#else
    return static_cast<int64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

} // namespace detail

// single thread counter
/// Accumulates timestamp samples for a single time point owned by a thread.
class ThreadPointAccumulator {
public:
    uint64_t startTimestamp_{0};
    uint64_t accumulatedTicks_{0};
    uint64_t sampleCount_{0};

    // reset 
    void reset() {
        sampleCount_ = 0;
        accumulatedTicks_ = 0;
    }

    // start to measure
    void start() {
        startTimestamp_ = detail::TimestampReader::readStart();
    }

    // stop the measure
    void stop() {
        accumulatedTicks_ += (detail::TimestampReader::readStop() - startTimestamp_);
        sampleCount_++;
    }

    // return true if point is used
    bool used() const {
        return sampleCount_ > 0;
    }

    // get collected delta sum of time
    uint64_t getDeltaSum() const {
        return accumulatedTicks_;
    }
    
    // get counter. Number of measures
    uint64_t getCounter() const {
        return sampleCount_;
    }

    // print information about this time point
    void print(std::ostream &stream, double tickTimeNs) const {
        double meanNs = 0;
        if (sampleCount_ > 0) {
            meanNs = (accumulatedTicks_ * tickTimeNs) / sampleCount_;
        }
        stream
            << "deltaSum=" << accumulatedTicks_ 
            << ", counter=" << sampleCount_
            << ", deltaSumNs=" << static_cast<uint64_t>(accumulatedTicks_ * tickTimeNs)
            << ", meanNs=" << static_cast<uint64_t>(meanNs);
    }
};

// additional info for Time point
/// Stores human readable name and parent linkage for a time point.
class TimePointMetadata {
public:
    void setName(const char* name) {
        if (name != nullptr) {
            strncpy(name_.data(), name, name_.size()-1);
            name_.data()[name_.size()-1] = '\0';
        }
    }

    // print to stream
    void print(std::ostream &stream) {
        stream << "[" << name_.data() << "]";
    }

    void setUpperLevel(int level) {
        upperLevel_ = level;
    }
    std::array<char, 64> name_{'\0'};
    int upperLevel_{-1};  // id of upper level counter 
};

/// Measures and caches CPU frequency to translate raw timestamp ticks
/// into nanoseconds. On non-TSC platforms this effectively acts as a
/// pass-through that treats the steady clock ticks as nanoseconds.
class CpuFrequencyCalibrator {
public:
    // measure current CPU frequency
    double measureCPUFreqMHz(size_t calibrationPeriodMs = 10) {
        using namespace std::chrono;

        const auto targetDuration = std::max(milliseconds(calibrationPeriodMs), milliseconds(1));
        const auto startWall = steady_clock::now();
        const uint64_t startTicks = detail::TimestampReader::readStart();
        const auto deadline = startWall + targetDuration;

        while (steady_clock::now() < deadline) {
            std::this_thread::yield();
        }

        const uint64_t endTicks = detail::TimestampReader::readStop();
        const auto endWall = steady_clock::now();
        const auto elapsedWall = duration_cast<nanoseconds>(endWall - startWall);

        const uint64_t deltaTicks = endTicks > startTicks ? (endTicks - startTicks) : 0;
        if (deltaTicks == 0 || elapsedWall.count() <= 0) {
            return cpuFreqMHz;
        }

        const double tickTimeNs = static_cast<double>(elapsedWall.count()) / static_cast<double>(deltaTicks);
        const double freqMHz = 1000.0 / tickTimeNs;
        return freqMHz;
    }

    // recalibrate
    void recalibrateCPUFreq() {
        auto freq = measureCPUFreqMHz();
        setCPUFreqMHz(freq);
    }

    // set CPU frequency. If you know frequency of you CPU
    void setCPUFreqMHz(double freqMHz) {
        cpuFreqMHz = freqMHz;
        tickTimeNs_ = 1000.0 / freqMHz;
    }

    double getTickTimeNs() const {
        return tickTimeNs_;
    }

    double getCPUFreqMHz() const {
        return cpuFreqMHz;
    }

private:
    static constexpr double defaultCpuFreqMhz_ = 2600;  // default value for 2.6 GHz CPU
    double cpuFreqMHz = defaultCpuFreqMhz_;
    double tickTimeNs_ = 1000.0 / defaultCpuFreqMhz_;  // time of one tick in nanoseconds
};



/// Global singleton that aggregates timing information from all threads.
/// Each counter slot is thread-safe and can be sampled concurrently.
class ExecutionProfiler {
public:
    static constexpr size_t kCounterWidth = 24; // width of counter in bits, so deltaTime will be (64 - kCounterWidth) bit width
    static constexpr size_t kNumCounters = 64;  // number of entries for counters

    // for counter width in 24 bits:
    //     max counter value (number of measurements) is 2^24 = 16'777'216
    //     max sum of CPU ticks 2^40 = 1'099'511'627'776 => (for 2.6 GHz) => 422'889'087'606 ns => 422 sec

    // -----------------------------------------
    static_assert(kCounterWidth >= 1 && kCounterWidth <= 63);
    static constexpr size_t kDeltaSumWidth = 64 - kCounterWidth;
    static constexpr uint64_t counterMask = static_cast<uint64_t>(-1) << kDeltaSumWidth;
    static constexpr uint64_t deltaSumMask = static_cast<uint64_t>(-1) >> kCounterWidth;

    // returns maximum possible value for counter
    static uint64_t getMaxCounterValue() {
        return (1ull << kCounterWidth) - 1;
    }

    // returns maximum possible value for deltasum of cpu ticks
    static uint64_t getMaxDeltaSumValue() {
        return (1ull << kDeltaSumWidth) - 1;
    }

    // Thread-local bookkeeping describing an in-flight measurement.
    struct ThreadPointState {
        uint64_t startTick_{0};
        bool isActive_{false};
    };

    // used to decode mixed value
    struct DecodedStatistics {
        uint64_t counter_;
        uint64_t deltaSumTicks_;
    };

    // 
    struct CounterSnapshot {
        uint64_t counter_;
        uint64_t deltaSumTicks_;
        uint64_t deltaSumNs_;
        uint64_t meanTimeNs_;

        void set(DecodedStatistics &counterDelta, double tickTimeNs) {
            counter_ = counterDelta.counter_;
            deltaSumTicks_ = counterDelta.deltaSumTicks_;
            deltaSumNs_ = static_cast<uint64_t>(static_cast<double>(deltaSumTicks_) * tickTimeNs);
            if (counter_ == 0) {
                meanTimeNs_ = 0;
                return;
            }
            double meanTimeTicks = static_cast<double>(counterDelta.deltaSumTicks_) / static_cast<double>(counter_);
            meanTimeNs_= static_cast<uint64_t>(meanTimeTicks * tickTimeNs);
        }
    };

    /// Aggregated, lock-free statistics shared across all threads for a
    /// particular counter slot.
    struct GlobalPointStatistics {
        std::atomic<uint64_t> packedStatistics_{0};  // mixed, combines counter and time delta

        // void addDelta(uint64_t delta) {
        //     auto a = (1ull << kCounterWidth) | delta;
        //     packedStatistics_.fetch_add(a); 
        // }

        static uint64_t getCounter(uint64_t mixed) {
            return (mixed & counterMask) >> kDeltaSumWidth;
        }

        ///
        ///
        ///
        static uint64_t getDeltaSumTicks(uint64_t mixed) {
            return (mixed & deltaSumMask);
        }

        static DecodedStatistics getInfoTick(uint64_t mixed) {
            DecodedStatistics result;
            result.counter_ = getCounter(mixed);
            result.deltaSumTicks_ = getDeltaSumTicks(mixed);
            return result;
        }

        DecodedStatistics getInfoTick() const {
            uint64_t mixed = packedStatistics_.load(std::memory_order_relaxed);
            DecodedStatistics result;
            result.counter_ = getCounter(mixed);
            result.deltaSumTicks_ = getDeltaSumTicks(mixed);
            return result;
        }

        CounterSnapshot getInfo(double tickTimeNs) const {
            auto infoTicks = getInfoTick();
            CounterSnapshot result;
            result.set(infoTicks, tickTimeNs);
            return result;
        }

        void reset() {
            packedStatistics_.store(0, std::memory_order_relaxed);
        }

        // Adds a new duration sample while protecting against overflow.
        void accumulate(uint64_t deltaTicks) {
            const uint64_t maxDelta = ExecutionProfiler::getMaxDeltaSumValue();
            const uint64_t clampedDelta = std::min(deltaTicks, maxDelta);

            uint64_t current = packedStatistics_.load(std::memory_order_relaxed);
            while (true) {
                const uint64_t currentCounter = getCounter(current);
                const uint64_t currentDelta = getDeltaSumTicks(current);

                const bool counterSaturated = currentCounter >= ExecutionProfiler::getMaxCounterValue();
                const uint64_t nextCounter = counterSaturated ? currentCounter : (currentCounter + 1);

                uint64_t nextDelta = currentDelta + clampedDelta;
                if (nextDelta > maxDelta) {
                    nextDelta = maxDelta;
                }

                const uint64_t next = (nextCounter << ExecutionProfiler::kDeltaSumWidth) | nextDelta;

                if (packedStatistics_.compare_exchange_weak(current, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            }
        }
    };

    // Singleton instance for the process.
    static ExecutionProfiler* instance() {
        static ExecutionProfiler mainProfiler;
        return &mainProfiler;
    }

    void startPoint(size_t id) {
        if (id < kNumCounters) {
            if (not threadLocalStates_[id].isActive_) {
                threadLocalStates_[id].isActive_ = true;
                threadLocalStates_[id].startTick_ = detail::TimestampReader::readStart();
            }
        }
    }

    void stopPoint(size_t id) {
        if (id < kNumCounters) {
            if (threadLocalStates_[id].isActive_) {
                auto delta = detail::TimestampReader::readStop() - threadLocalStates_[id].startTick_;
                globalStatistics_[id].accumulate(delta);
                threadLocalStates_[id].isActive_ = false;
            }
        }
    }

    void setName(size_t id, const char* name) {
        if (id < kNumCounters) {
            pointMetadata_[id].setName(name);
        }
    }

    void reset(size_t id) {
        if (id < kNumCounters) {
            globalStatistics_[id].reset();
        }
    }

    void resetAll() {
        for (auto &stat : globalStatistics_) {
            stat.reset();
        }
    }

    // local start points 
    static thread_local std::array<ThreadPointState, kNumCounters> threadLocalStates_;
    std::array<GlobalPointStatistics, kNumCounters>  globalStatistics_;  // shared across all threads
    std::array<TimePointMetadata, kNumCounters>  pointMetadata_;  // human readable labels

    // process counters and deltas
    void printInfo(std::ostream &stream) {
        auto tickTimeNs = cpuFreqCalib.getTickTimeNs();
        for( size_t id = 0; id < kNumCounters; id++) {
            uint64_t raw = globalStatistics_[id].packedStatistics_.load(std::memory_order_relaxed);
            if (raw != 0) {
                uint64_t c = GlobalPointStatistics::getCounter(raw);
                uint64_t d = GlobalPointStatistics::getDeltaSumTicks(raw);
                // int timeNs = static_cast<int>((static_cast<double>(d) / static_cast<double>(c)) * tickTimeNs);

                // auto timeNs =  globalStatistics_[id].getMeanTimeNs(tickTimeNs);
                // stream << "Timepoint " << id << ": [" << pointMetadata_[id].name_.data() << "]" << " counter=" << c << ", deltaSum=" << d << ", timeNs=" << timeNs << "\n";
                auto info = globalStatistics_[id].getInfo(tickTimeNs);
                auto name = pointMetadata_[id].name_.data();
                stream << "Timepoint " << id << ": [" << name << "]" << " counter=" << info.counter_
                       << ", deltaSum=" << info.deltaSumTicks_ << ", timeNs=" << info.meanTimeNs_ << "\n";

            }
        }
    }
    // process counters and deltas
    void printInfoById(int id, std::ostream &stream) {
        auto tickTimeNs = cpuFreqCalib.getTickTimeNs();
        if (id < kNumCounters) {
            uint64_t raw = globalStatistics_[id].packedStatistics_.load(std::memory_order_relaxed);
            if (raw != 0) {
                uint64_t c = GlobalPointStatistics::getCounter(raw);
                uint64_t d = GlobalPointStatistics::getDeltaSumTicks(raw);
                // int timeNs = static_cast<int>((static_cast<double>(d) / static_cast<double>(c)) * tickTimeNs);

                // auto timeNs =  globalStatistics_[id].getMeanTimeNs(tickTimeNs);
                // stream << "Timepoint " << id << ": [" << pointMetadata_[id].name_.data() << "]" << " counter=" << c << ", deltaSum=" << d << ", timeNs=" << timeNs << "\n";
                auto info = globalStatistics_[id].getInfo(tickTimeNs);
                auto name = pointMetadata_[id].name_.data();
                stream << "Timepoint " << id << ": [" << name << "]" << " counter=" << info.counter_
                       << ", deltaSum=" << info.deltaSumTicks_ << ", timeNs=" << info.meanTimeNs_ << "\n";

            }
        }
    }
    CounterSnapshot getInfo(int k) const {
        auto tickTimeNs = cpuFreqCalib.getTickTimeNs();
        return globalStatistics_[k].getInfo(tickTimeNs);
    }

    void recalibrateCPUFreq() {
        cpuFreqCalib.recalibrateCPUFreq();
    }

    void setCPUFreqMHz(double freqMHz) {
        cpuFreqCalib.setCPUFreqMHz(freqMHz);
    }

    double getCPUFreqMHz() const {
        return  cpuFreqCalib.getCPUFreqMHz();
    }

    double getTickTimeNs() const {
        return cpuFreqCalib.getTickTimeNs();
        // return tickTimeNs_;
    }
private:
    CpuFrequencyCalibrator cpuFreqCalib;
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);
inline thread_local std::array<ExecutionProfiler::ThreadPointState, ExecutionProfiler::kNumCounters> ExecutionProfiler::threadLocalStates_;

// -----------------------------------------------------------------
// ----------------------   Counters for each thread  ----------------------


// Per-thread helper used when callers want isolated counters
// without touching the global singleton.
class ExecutionProfilerThread {
public:
    ExecutionProfilerThread() {
        processId_ = getpid();
        threadId_ = detail::queryThreadId();
        printIntervalRemaining_ = 1000;
    };
    static ExecutionProfilerThread* instance() { 
        static thread_local ExecutionProfilerThread executionProfiler;
        return &executionProfiler;
    }

    bool needPrint() {
        if (printIntervalRemaining_ <= 0) {
            printIntervalRemaining_ = 1000;
            return true;
        }
        return false;
    }

    void printTimePoint(std::ostream &stream, int pointID) {
        if (timePointAccumulators_[pointID].used()) {
            stream << "Timepoint " << pointID << ": ";
            timePointMetadata_[pointID].print(stream);
            stream << " ";
            timePointAccumulators_[pointID].print(stream, cpuFreqCalib.getTickTimeNs());  // TODO. change to <<
            stream << "\n";
        }
    }

    void printLowPoints(std::ostream &stream, int currLevelID, int indentLevel) {
        for (size_t currPointID = 0; currPointID < kTimePointArraySize_; currPointID++) {
            if (timePointAccumulators_[currPointID].used() && timePointMetadata_[currPointID].upperLevel_ == currLevelID) {
                if (indentLevel > 0) {
                    stream << std::setw(indentLevel * 4) << " ";
                }
                printTimePoint(stream, currPointID);
                printLowPoints(stream, currPointID, indentLevel + 1);
            }
        }
    }

    void printInfo(std::ostream &stream) {
        stream << "------------ Profiler info -----------\n";
        stream << "pid=" << processId_ << "|" << threadId_ << "\n";
        printLowPoints(stream, -1, 0);
        stream << "--------------------------------------\n";
    }

    void resetTimePoint(uint32_t pointID) {
        if (pointID < kTimePointArraySize_) {
            timePointAccumulators_[pointID].reset();
        }
    }

    void startTimePoint(uint32_t pointID) {
        if (pointID < kTimePointArraySize_) {
            timePointAccumulators_[pointID].start();
        }
    }

    void stopTimePoint(uint32_t pointID) {
        if (pointID < kTimePointArraySize_) {
            timePointAccumulators_[pointID].stop();
        }
        printIntervalRemaining_--;
    }

    void setName(uint32_t pointID, const char* name, int upperLevel = -1) {
        if (pointID < kTimePointArraySize_) {
            timePointMetadata_[pointID].setName(name);
            timePointMetadata_[pointID].setUpperLevel(upperLevel);
            timePointAccumulators_[pointID].reset();
        }
    }

    pid_t processId_{0};
    int64_t threadId_{0};

    // --------------- cpu freq ---------------
    void recalibrateCPUFreq() {
        cpuFreqCalib.recalibrateCPUFreq();
    }

    void setCPUFreqMHz(double freqMHz) {
        cpuFreqCalib.setCPUFreqMHz(freqMHz);
    }

    double getCPUFreqMHz() const {
        return  cpuFreqCalib.getCPUFreqMHz();
    }

    double getTickTimeNs() const {
        return cpuFreqCalib.getTickTimeNs();
    }

    CpuFrequencyCalibrator cpuFreqCalib;
    // --------------- cpu freq ---------------


    static const size_t kTimePointArraySize_ = 64;
    std::array<ThreadPointAccumulator, kTimePointArraySize_> timePointAccumulators_;
    std::array<TimePointMetadata, kTimePointArraySize_> timePointMetadata_;

    int printIntervalRemaining_{1000};
};

// Helper to profile a scope automatically. Construct it at the top of
// a scope to start the counter and let the destructor stop it even if
// the scope exits early via exceptions or returns.
class ScopedProfile {
public:
    explicit ScopedProfile(size_t counterId, bool enabled = true)
        : counterId_(counterId), enabled_(enabled) {
        if (enabled_) {
            ExecutionProfiler::instance()->startPoint(counterId_);
        }
    }

    ScopedProfile(const ScopedProfile&) = delete;
    ScopedProfile& operator=(const ScopedProfile&) = delete;

    ScopedProfile(ScopedProfile&& other) noexcept
        : counterId_(other.counterId_), enabled_(other.enabled_) {
        other.enabled_ = false;
    }

    ScopedProfile& operator=(ScopedProfile&& other) noexcept {
        if (this != &other) {
            if (enabled_) {
                ExecutionProfiler::instance()->stopPoint(counterId_);
            }
            counterId_ = other.counterId_;
            enabled_ = other.enabled_;
            other.enabled_ = false;
        }
        return *this;
    }

    ~ScopedProfile() {
        if (enabled_) {
            ExecutionProfiler::instance()->stopPoint(counterId_);
        }
    }

private:
    size_t counterId_;
    bool enabled_;
};

}
