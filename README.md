# ExecutionProfiler

Tiny header-only instrumentation profiler that collects high-resolution timing for up to 64 counters across multiple threads. The profiler exposes manual start/stop APIs, a scoped RAII helper, and per-thread views, making it easy to instrument hot paths without pulling in a heavyweight tracing dependency.

## Features
- Thread-safe global singleton with 64 configurable counter slots
- RAII-style `ScopedProfile` guard to instrument scopes safely
- Per-thread profiler (`ExecutionProfilerThread`) for isolated measurements
- Automatic CPU frequency calibration with fallback to steady clock when TSC is unavailable
- Saturating, lock-free aggregation to prevent counter wraparound
- Portable thread ID querying for Linux and macOS

## Getting Started
1. Drop `ExecutionProfiler.h` into your project include path.
2. Include it where you want to profile and set up a counter name:
   ```cpp
   #include "ExecutionProfiler.h"

   constexpr size_t kMyCounter = 0;

   void doWork() {
       profiler::ExecutionProfiler::instance()->setName(kMyCounter, "doWork");
       profiler::ScopedProfile scope(kMyCounter);
       // ... your work here ...
   }
   ```
3. For manual control you can call `startPoint(counterId)` / `stopPoint(counterId)` directly.
4. When you need to reset statistics, use `reset(counterId)` or `resetAll()`.

## CLI Utilities
Two small helper programs are provided for validation and benchmarking:

### Functional tests
```
clang++ -std=c++17 profiler_test.cpp -o profiler_test -pthread
./profiler_test
```
The binary runs a suite of checks covering scoped profiling, manual start/stop, disabled scopes, move semantics, and parallel aggregation. All tests should print `PASS`.

### Overhead measurement
```
clang++ -std=c++17 profiler_overhead.cpp -o profiler_overhead -pthread
./profiler_overhead
```
This microbenchmark compares an empty loop against `startPoint/stopPoint` and `ScopedProfile`. Subtracting the baseline shows the profiler’s per-call overhead on your machine.

## Notes on Timing Sources
- On x86 targets the profiler serializes `rdtsc` / `rdtscp` with LFENCE to pick up stable cycle counts.
- On architectures without TSC (e.g., Apple silicon) the profiler falls back to `std::chrono::steady_clock`, so raw overhead will be higher, but results remain monotonic and portable.

## License
MIT
