// A recomp module on the Brick: PSXRecomp's runtime keeps its own game, sound and pad (SDL2), while its frames
// go to the Vulkan presenter (video.h: fb0's pages, as Deadstation's own). SDL has no
// video here (PSXRecomp patch 0010), so that it never takes fb0. The runtime calls psx_host_present (patch 0007) with each
// frame's active display rect, ARGB8888 rows.
#include "host_link.h"
#include "log.h"
#include "psx_gpu.h"
#include "replay.h"
#include "video.h"

#include <SDL.h>
#include <sched.h>
#include <sys/klog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <map>
#include <execinfo.h>
#include <thread>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// The module's log (stdout and stderr: a file on the SD card) written on a thread of its own: a write to the card can stall
// for a tenth of a second, which on the game's thread is a dropped frame. Everything printed goes through a pipe, drained
// at once on another core; DEADSTATION_LOG_DIRECT=1 writes directly.
namespace
{
int logTarget = -1;
void logDrain(int from)
{
	char buffer[4096];
	for (;;)
	{
		const ssize_t n = read(from, buffer, sizeof(buffer));
		if (n < 0 && errno == EINTR)
			continue;
		if (n <= 0)
			return;
		for (ssize_t done = 0; done < n;)
		{
			const ssize_t w = write(logTarget, buffer + done, (size_t)(n - done));
			if (w <= 0)
				break;
			done += w;
		}
	}
}
struct LogThread
{
	LogThread()
	{
		if (getenv("DEADSTATION_LOG_DIRECT") != nullptr)
			return;
		int fds[2];
		if (pipe(fds) != 0)
			return;
		logTarget = dup(1);
		if (logTarget < 0)
			return;
		setvbuf(stdout, nullptr, _IOLBF, 0);
		dup2(fds[1], 1);
		dup2(fds[1], 2);
		close(fds[1]);
		std::thread(logDrain, fds[0]).detach();
		atexit([] { fflush(stdout); usleep(30000); });
	}
} logThread;
}

extern "C" void psx_host_set_frame_period_ms(double period_ms);	// patch 0027
// the panel's frame in milliseconds, once the screen's up (a module's plugin that keeps real time reads it)
extern "C" { double deadstation_frame_period_ms = 1000.0 / 60.0; }

// frames the game's own logic finished, counted by a module's plugin at the game's frame end (0: this module doesn't)
extern "C" { uint32_t deadstation_game_frames = 0; }

namespace
{
// never destroyed: at exit the Vulkan driver may be gone before static destructors run
Video& video = *new Video;
VideoLayer *moduleLayer = nullptr;	// (deadstation_module_layer)
bool tried = false;
// DEADSTATION_PERF: a line a second, frames shown, the longest gap between two, the time spent waiting for a page
bool perf = false;
int consoleLevelWas = 0;		// the kernel's console level, to put back
std::chrono::steady_clock::time_point perfStart, lastPresent;
unsigned perfFrames = 0;
double perfLongest = 0, perfPresenting = 0, perfSubmitting = 0;
// DEADSTATION_PERF: the module's thread's CPU clock (taken at start-up, on it): its share of a core each second
clockid_t moduleClock;
bool moduleClockOk = false;
double moduleCpuLast = 0;
// the PlayStation's clock: guest time run against wall time (100%: the game at its own speed)
extern "C" uint64_t psx_cycle_count;
uint64_t guestCyclesLast = 0;
// frames the game drew (a frame shown with none of its drawing since the last: the game repeated it): the game's own rate,
// in the perf line and, with Show FPS on, in the corner as the emulator's counter
unsigned perfGameFrames = 0, fpsGameFrames = 0;
uint32_t perfBuiltLast = 0, fpsBuiltLast = 0;
bool showFps = false, fpsShown = false;
std::atomic<bool> fpsHeld { false };	// off for the menu
std::chrono::steady_clock::time_point fpsStart;
// the frame counter's characters, drawn by the emulator (a module has no fonts), as tools make them: their padding and height
// and how many, then for each its code, its width and its pixels (BGRA, at the panel's size). All 32-bit values.
struct CounterCell
{
	uint32_t width = 0;
	std::vector<uint32_t> pixels;
};
std::map<char, CounterCell> counterCells;
uint32_t cellPad = 0, cellHeight = 0;

// text in those characters, at the size the screen is in: it looks the same however large the frame is
Panel::Corner counterCorner(const char *text, int screenWidth)
{
	Panel::Corner corner;
	uint32_t width = cellPad * 2;
	for (const char *c = text; *c != '\0'; c++)
		if (const auto cell = counterCells.find(*c); cell != counterCells.end())
			width += cell->second.width;
	const uint32_t height = cellHeight + cellPad * 2;
	std::vector<uint32_t> full((size_t)width * height, Panel::Corner::Box);
	uint32_t x = cellPad;
	for (const char *c = text; *c != '\0'; c++)
	{
		const auto cell = counterCells.find(*c);
		if (cell == counterCells.end())
			continue;
		for (uint32_t y = 0; y < cellHeight; y++)
			memcpy(&full[(size_t)(y + cellPad) * width + x], &cell->second.pixels[(size_t)y * cell->second.width], cell->second.width * 4);
		x += cell->second.width;
	}
	corner.width = (int)width * std::min(screenWidth, 1024) / 1024;
	corner.height = (int)height * std::min(screenWidth, 1024) / 1024;
	corner.pixels.resize((size_t)corner.width * corner.height);
	for (int y = 0; y < corner.height; y++)
		for (int cx = 0; cx < corner.width; cx++)
			corner.pixels[(size_t)y * corner.width + cx] = full[(size_t)(y * (int)height / corner.height) * width + cx * (int)width / corner.width];
	return corner;
}
double moduleCpu()
{
	timespec t {};
	return moduleClockOk && clock_gettime(moduleClock, &t) == 0 ? t.tv_sec + t.tv_nsec * 1e-9 : 0;
}
// SIGUSR2 (PSXRecomp takes SIGUSR1 as a quit): what the display shows, read back from fb0 uncached into /tmp/ds-shot.ppm (as Deadstation's own)
std::atomic<bool> shotWanted { false };
// DEADSTATION_PROFILE=FILE (tests): the program counter where the CPU time goes, sampled every millisecond of
// CPU time (SIGPROF), written to FILE at MENU's quit with the executable's load address, for symbolising on the PC
constexpr unsigned ProfileCap = 1u << 20;
uint64_t *profile = nullptr;
std::atomic<unsigned> profileCount { 0 };
// four words a sample: the program counter, the return address (its caller, for a leaf such as memcmp), the thread
// and the guest function last entered (the game's or the SDK's, by address)
extern "C" volatile uint32_t g_psx_last_fn_entry;
void profileSample(int, siginfo_t *, void *context)
{
	const unsigned i = profileCount.fetch_add(1, std::memory_order_relaxed);
	if (i < ProfileCap / 4)
	{
		const ucontext_t *uc = (const ucontext_t *)context;
		profile[4 * i] = uc->uc_mcontext.pc;
		profile[4 * i + 1] = uc->uc_mcontext.regs[30];
		profile[4 * i + 2] = (uint64_t)syscall(SYS_gettid);
		profile[4 * i + 3] = g_psx_last_fn_entry;
	}
}
void profileStart()
{
	if (getenv("DEADSTATION_PROFILE") == nullptr)
		return;
	profile = new uint64_t[ProfileCap];
	struct sigaction sa = {};
	sa.sa_sigaction = profileSample;
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sigaction(SIGPROF, &sa, nullptr);
	itimerval every = { { 0, 1000 }, { 0, 1000 } };
	setitimer(ITIMER_PROF, &every, nullptr);
}
void profileWrite()
{
	const char *path = getenv("DEADSTATION_PROFILE");
	if (profile == nullptr || path == nullptr)
		return;
	itimerval off = {};
	setitimer(ITIMER_PROF, &off, nullptr);
	FILE *f = fopen(path, "w");
	if (f == nullptr)
		return;
	// the code mappings (the executable's and the libraries'), to turn the samples into their own addresses
	if (FILE *maps = fopen("/proc/self/maps", "r"))
	{
		char line[512];
		while (fgets(line, sizeof(line), maps) != nullptr)
			if (strstr(line, " r-xp ") != nullptr || strstr(line, "_recomp_module") != nullptr)
				fprintf(f, "map %s", line);
		fclose(maps);
	}
	fprintf(f, "main %d\n", (int)getpid());
	const unsigned n = std::min(profileCount.load(), ProfileCap / 4);
	for (unsigned i = 0; i < n; i++)
		fprintf(f, "%llx %llx %llu %llx\n", (unsigned long long)profile[4 * i], (unsigned long long)profile[4 * i + 1],
				(unsigned long long)profile[4 * i + 2], (unsigned long long)profile[4 * i + 3]);
	fclose(f);
}
// a host crash: where (pc, lr, the faulting address, the executable's load address) into the log, then the
// runtime's own handler (installed after ours at its start, so chained from the first frame on)
struct sigaction previousSegv;
void onSegv(int sig, siginfo_t *info, void *context)
{
	const ucontext_t *uc = (const ucontext_t *)context;
	char line[256];
	uintptr_t base = 0;
	if (FILE *maps = fopen("/proc/self/maps", "r"))
	{
		char m[512];
		while (base == 0 && fgets(m, sizeof(m), maps) != nullptr)
			if (strstr(m, "_recomp_module") != nullptr)
				base = (uintptr_t)strtoull(m, nullptr, 16);
		fclose(maps);
	}
	const int n = snprintf(line, sizeof(line), "Deadstation: recomp module crash: signal %d pc %llx lr %llx address %llx base %llx\n",
			sig, (unsigned long long)uc->uc_mcontext.pc, (unsigned long long)uc->uc_mcontext.regs[30],
			(unsigned long long)(uintptr_t)info->si_addr, (unsigned long long)base);
	if (n > 0)
		(void)!write(1, line, (size_t)n);
	sigaction(SIGSEGV, &previousSegv, nullptr);
	raise(SIGSEGV);
}
int stopCode = 0;			// what the module exits with, the watchdog's too

// the end of the module, however it ends (Quit, exit): every Vulkan object destroyed, the GPU's first, then the scene
// layer's and the presenter's with its device, while the driver is still loaded (registered with atexit after the
// presenter starts, so it runs before the driver's own exit handlers); the driver otherwise keeps their memory (~25 MB
// a run). A watchdog ends the process if the GPU never goes idle.
void hostTeardown()
{
	static bool done = false;
	if (done)
		return;
	done = true;
	profileWrite();
	std::thread([] {
		std::this_thread::sleep_for(std::chrono::seconds(3));
		_exit(stopCode);
	}).detach();
	psx_gpu_stop();
	if (video.ready())
	{
		video.submitDrain();
		video.gpuDevice().device.waitIdle();
	}
	psx_gpu_release();
	if (consoleLevelWas > 0)
		klogctl(8, nullptr, consoleLevelWas), consoleLevelWas = 0;
	video.setLayer(nullptr);
	if (moduleLayer != nullptr)
		moduleLayer->term();
	video.term();
}

// before main. The runtime starts no SDL video with a host presenting (PSXRecomp patch 0010: the Brick's SDL2
// has no dummy video driver); stdout line by line, so a module's log survives a crash; SIGUSR2 from the start
// (its default would end the module)
// an exit() from anywhere: who called it (return addresses, and the executable's load address to symbolise them)
void logExit()
{
	void *frames[24];
	const int n = backtrace(frames, 24);
	uintptr_t base = 0;
	if (FILE *maps = fopen("/proc/self/maps", "r"))
	{
		char m[512];
		while (base == 0 && fgets(m, sizeof(m), maps) != nullptr)
			if (strstr(m, "_recomp_module") != nullptr)
				base = (uintptr_t)strtoull(m, nullptr, 16);
		fclose(maps);
	}
	fprintf(stdout, "Deadstation: recomp module exit, base %llx:", (unsigned long long)base);
	for (int i = 0; i < n; i++)
		fprintf(stdout, " %llx", (unsigned long long)(uintptr_t)frames[i]);
	fprintf(stdout, "\n");
	fflush(stdout);
}

__attribute__((constructor)) void brickStart()
{
	atexit(logExit);
	moduleClockOk = pthread_getcpuclockid(pthread_self(), &moduleClock) == 0;
	// GTE positions and depths for the GPU's triangles (psx_gpu.cpp: its native scene, on unless DEADSTATION_SCENE=0, and
	// DEADSTATION_GPU_DEPTH=1): PSXRecomp's geometry correction tracks each projected corner to the triangles that use it
	// (read before main)
	{
		const char *depth = getenv("DEADSTATION_GPU_DEPTH"), *scene = getenv("DEADSTATION_SCENE");
		if ((depth != nullptr && strcmp(depth, "1") == 0) || scene == nullptr || strcmp(scene, "0") != 0)
			setenv("PSX_GEOMETRY_CORRECTION", "1", 0);
	}
	setvbuf(stdout, nullptr, _IOLBF, 0);
	signal(SIGUSR2, [](int) { shotWanted = true; });
	profileStart();		// from the start: the module's boot too
}

// at exit: no Vulkan (the driver may be gone before static destructors run); the pak's launch.sh puts NextUI's display
// mode back
struct Ending
{
	~Ending()
	{
		video.setLayer(nullptr);
	}
} ending;
}

Video& hostVideo()
{
	if (!tried)
	{
		tried = true;
		perf = getenv("DEADSTATION_PERF") != nullptr;
		struct sigaction sa = {};
		sa.sa_sigaction = onSegv;
		sa.sa_flags = SA_SIGINFO;
		sigaction(SIGSEGV, &sa, &previousSegv);
		// Every interrupt on this device lands on core 0 (the SD card, Wi-Fi, the display, the GPU): the game's threads, all started from here on, keep to the other three.
		if (sysconf(_SC_NPROCESSORS_ONLN) >= 4)
		{
			cpu_set_t others;
			CPU_ZERO(&others);
			for (int core = 1; core < 4; core++)
				CPU_SET(core, &others);
			sched_setaffinity(0, sizeof(others), &others);
		}
		// This kernel prints all it logs to a serial port at 115200 baud, holding the console's lock, and a flip of the display (FBIOPAN_DISPLAY) takes that lock: a kilobyte of it (Wi-Fi
		// losing its access point and trying again, each try about that much) stops the frames for 90 ms. Off the port while a game runs; it's all in dmesg still.
		{
			int console = 0;
			if (FILE *f = fopen("/proc/sys/kernel/printk", "r"))
			{
				if (fscanf(f, "%d", &console) == 1 && console > 1 && klogctl(8, nullptr, 1) == 0)
					consoleLevelWas = console;
				fclose(f);
			}
		}
		// DEADSTATION_RESOLUTION=1024: fb0 at the panel's own size; 640 or 320: that, at 4:3, the display engine scaling it up (the module's Resolution setting picks it)
		const char *res = getenv("DEADSTATION_RESOLUTION");
		std::string error;
		moduleLayer = deadstation_module_layer != nullptr ? deadstation_module_layer() : nullptr;
		const bool layered = moduleLayer != nullptr;
		// frames up to 1280 x 1024: the runtime's internal resolution at 2x (PSX_INTERNAL_RESOLUTION, run.sh)
		// DEADSTATION_ASYNC_SUBMIT (comparisons): the driver's submissions on a thread of their own. Off: PowerVR's driver
		// holds one lock per process across a submission, so the module's other Vulkan calls wait for it anyway.
		video.setAsyncSubmit(getenv("DEADSTATION_ASYNC_SUBMIT") != nullptr);
		const bool ready = video.init(res != nullptr && strcmp(res, "1024") == 0, true, error, 1280, 1024, layered, res != nullptr && strcmp(res, "320") == 0 ? 320 : 640);
		if (ready)
			atexit(hostTeardown);
		// the panel at the game's rate (Panel::matchRate), the frames paced at the panel's own (DEADSTATION_PANEL_PACING=0: neither, the PlayStation's 59.94 Hz, a frame repeated
		// every few seconds on this panel). Exactly its rate: any faster and the game itself runs fast, with frames never shown
		const char *pacing = getenv("DEADSTATION_PANEL_PACING");
		if (ready && (pacing == nullptr || strcmp(pacing, "0") != 0))
		{
			video.panel().matchRate();
			deadstation_frame_period_ms = 1000.0 / video.panel().refreshRate();
			psx_host_set_frame_period_ms(deadstation_frame_period_ms);
		}
		if (!ready)
			DS_LOG("recomp module: no screen: %s", error.c_str());
		else if (layered)
		{
			if (moduleLayer->init(video.layerTarget(), error))
				video.setLayer(moduleLayer);
			else
				DS_LOG("recomp module: the module's layer failed: %s", error.c_str());
		}
	}
	return video;
}

// after each frame shown (from the CPU's pixels here, or the GPU's image in psx_gpu.cpp)
void hostFrameShown(int width, int height, bool newFrame)
{
	perfGameFrames += newFrame, fpsGameFrames += newFrame;
	if (showFps && !fpsHeld)
	{
		const auto now = std::chrono::steady_clock::now();
		if (fpsStart.time_since_epoch().count() == 0)
			fpsStart = now;
		const double span = std::chrono::duration<double>(now - fpsStart).count();
		if (span >= 1.0)
		{
			char text[16];
			// the game's own count where its plugin keeps one, else the frames it drew
			const uint32_t built = deadstation_game_frames;
			snprintf(text, sizeof(text), "%4.1f", (built != fpsBuiltLast ? built - fpsBuiltLast : fpsGameFrames) / span);
			fpsBuiltLast = built;
			if (!counterCells.empty())
			{
				std::vector<Panel::Corner> corners;
				corners.push_back(counterCorner(text, (int)video.panel().extent().width));
				video.panel().setCorners(std::move(corners));
				fpsShown = true;
			}
			fpsGameFrames = 0;
			fpsStart = now;
		}
	}
	if (perf)
	{
		const auto now = std::chrono::steady_clock::now();
		static const auto started = now;
		static int shownW = 0, shownH = 0;
		if (width != shownW || height != shownH)
		{
			shownW = width, shownH = height;
			DS_LOG("recomp timeline: %.1f s: %d x %d on screen", std::chrono::duration<double>(now - started).count(), width, height);
		}
		if (perfFrames == 0 && perfStart.time_since_epoch().count() == 0)
			perfStart = lastPresent = now;
		perfLongest = std::max(perfLongest, std::chrono::duration<double>(now - lastPresent).count());
		perfPresenting += video.takeWaited();
		perfSubmitting += video.takeSubmitting();
		lastPresent = now;
		perfFrames++;
		const double span = std::chrono::duration<double>(now - perfStart).count();
		if (span >= 1.0)
		{
			const double cpu = moduleCpu();
			const uint64_t guest = psx_cycle_count;
			const uint32_t built = deadstation_game_frames;
			DS_LOG("recomp perf: the game ran %.1f frames/s, drew %.1f (%.1f shown), longest gap %.1f ms, waiting for a page %.1f ms a frame, "
					"submitting %.1f ms a frame, %dx%d; the module's thread %.0f%% of a core; guest speed %.1f%%; %u in-between frames", (built - perfBuiltLast) / span,
					perfGameFrames / span, perfFrames / span, perfLongest * 1000,
					perfPresenting * 1000 / perfFrames, perfSubmitting * 1000 / perfFrames, width, height,
					(cpu - moduleCpuLast) * 100 / span, guestCyclesLast ? (double)(guest - guestCyclesLast) / (33868800.0 * span) * 100 : 0.0,
					deadstation_replay_take_count());
			moduleCpuLast = cpu;
			guestCyclesLast = guest;
			perfBuiltLast = built;
			perfStart = now;
			perfFrames = perfGameFrames = 0;
			perfLongest = perfPresenting = perfSubmitting = 0;
		}
	}
	if (shotWanted.exchange(false))
	{
		std::vector<uint8_t> rgb;
		int w = 0, h = 0;
		video.panel().drain();
		if (video.panel().readShown(rgb, w, h))
		{
			std::ofstream out("/tmp/ds-shot.ppm.tmp", std::ios::binary);
			out << "P6\n" << w << " " << h << "\n255\n";
			out.write((const char *)rgb.data(), (std::streamsize)rgb.size());
			out.close();
			std::rename("/tmp/ds-shot.ppm.tmp", "/tmp/ds-shot.ppm");
		}
	}
}

void hostInput()
{
	hostLinkPoll();
}

void hostStop(int code)
{
	// out at once: the runtime's threads keep running through exit()'s destructors (a crash there), its memory cards are on
	// the SD card already (each sector as written), and the pak's launch.sh puts NextUI's display mode back (the watchdog
	// in hostTeardown: waiting here for the GPU to go idle could hang the quit)
	stopCode = code;
	fflush(stdout);
	usleep(30000);	// (the log's thread has its last lines)
	hostTeardown();
}

void hostShowFps(bool on)
{
	showFps = on;
	fpsStart = {};
	fpsGameFrames = 0;
	if (!on && fpsShown)
	{
		video.panel().setCorners({});
		fpsShown = false;
	}
}

void hostCounterCells(const std::string& file)
{
	counterCells.clear();
	FILE *f = fopen(file.c_str(), "rb");
	if (f == nullptr)
	{
		DS_LOG("recomp module: no counter characters at %s", file.c_str());
		return;
	}
	uint32_t head[3] = {};
	if (fread(head, sizeof(head[0]), 3, f) == 3)
	{
		cellPad = head[0];
		cellHeight = head[1];
		for (uint32_t i = 0; i < head[2]; i++)
		{
			uint32_t cell[2] = {};
			if (fread(cell, sizeof(cell[0]), 2, f) != 2)
				break;
			CounterCell made;
			made.width = cell[1];
			made.pixels.resize((size_t)cell[1] * cellHeight);
			if (fread(made.pixels.data(), sizeof(uint32_t), made.pixels.size(), f) != made.pixels.size())
				break;
			counterCells[(char)cell[0]] = std::move(made);
		}
	}
	fclose(f);
}

bool hostHideFps()
{
	fpsHeld = true;
	if (!fpsShown)
		return false;
	video.panel().setCorners({});
	fpsShown = false;
	return true;
}

void hostMenuClosed()
{
	const auto now = std::chrono::steady_clock::now();
	const uint32_t built = deadstation_game_frames;
	fpsHeld = false;
	fpsStart = now;
	fpsGameFrames = 0;
	fpsBuiltLast = built;
	perfStart = lastPresent = now;
	perfFrames = perfGameFrames = 0;
	perfLongest = perfPresenting = perfSubmitting = 0;
	(void)video.takeWaited();
	(void)video.takeSubmitting();
	moduleCpuLast = moduleCpu();
	guestCyclesLast = psx_cycle_count;
	perfBuiltLast = built;
}

// the software renderer's frames (PSXRecomp's renderer = software)
extern "C" int psx_host_present(const uint32_t *pixels, int width, int height)
{
	Video& v = hostVideo();
	if (!v.ready() || pixels == nullptr || width <= 0 || height <= 0)
		return v.ready() ? 1 : 0;
	v.frame(pixels, (unsigned)width, (unsigned)height, (size_t)width * 4, Video::Format::XRGB8888);
	hostFrameShown(width, height, true);
	hostInput();
	return 1;
}
