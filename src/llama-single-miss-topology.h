#pragma once

#ifdef __linux__
#include <fstream>
#include <sched.h>
#include <set>
#include <string>
#endif

inline int llama_single_miss_physical_cores() {
#ifdef __linux__
    cpu_set_t cpus;
    if (sched_getaffinity(0, sizeof(cpus), &cpus) != 0) {
        return 0;
    }
    std::set<std::string> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &cpus)) {
            std::ifstream topology("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
            std::string siblings;
            if (!(topology >> siblings)) {
                return 0;
            }
            cores.insert(siblings);
        }
    }
    return int(cores.size());
#else
    return 0;
#endif
}
