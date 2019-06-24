#ifndef STOPWATCH_H__
#define STOPWATCH_H__

#include <chrono>
#include <iostream>
#include <SafeCall.h>

struct StopWatch {
	static void Tick(const std::string progName) {
		t1 = std::chrono::system_clock::now();
	}

	static void Tock(const std::string progName) {

		SafeCall(cudaDeviceSynchronize());
		auto t2 = std::chrono::system_clock::now();
		auto result = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

		std::cout << "Process " << progName << " Finished in " << result.count() << " ms" << std::endl;
	}

	static bool enabled;
	static std::chrono::system_clock::time_point t1;
};

std::chrono::system_clock::time_point StopWatch::t1;

#endif
