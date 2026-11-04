// A recomp module's link to the emulator that hosts it, and how it starts. Deadstation owns the MENU button, the menu,
// Options and the store; the module plays the game and does as it is told. Two pipes, DEADUI_HOST=<read fd>,<write fd>,
// lines of text:
//   from Deadstation: pause, resume, set <key> <value>, quit
//   to Deadstation:   ready (the first frame shown), paused (nothing more of this process reaches the screen)
// Paused, the module's thread stays here, so the guest is held where it was (its clock, its game logic), the render thread
// has nothing to do and the sound is held: the screen is Deadstation's until resume. Run without Deadstation, the game simply
// plays, with no menu.
//
// Deadstation starts a module as <exec> <disc> [--set key=value]...: PSXRecomp's own arguments are made here. The module's
// files are all in $DEADSTATION_HOME/recomp-data/<id>/, the runtime's folder as it sees it; what it only reads stays beside
// the executable.
#include "host_link.h"
#include "log.h"
#include "psx_gpu.h"
#include "video.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

extern "C" void psx_host_audio_hold(int held);	// patch 0029
extern "C" void psx_host_reload_input(void);	// patch 0031
extern "C" int __real_main(int argc, char **argv);

namespace fs = std::filesystem;

namespace
{

constexpr const char *ControlPrefix = "control.";

// the Brick's inputs by their labels, as PSXRecomp's input.ini names them (SDL's names go by a button's place on the pad)
const std::map<std::string, std::string> Sources {
	{ "A", "b" }, { "B", "a" }, { "X", "y" }, { "Y", "x" }, { "L1", "leftshoulder" }, { "R1", "rightshoulder" },
	{ "L2", "lefttrigger" }, { "R2", "righttrigger" }, { "SELECT", "back" }, { "START", "start" }, { "FN1", "leftstick" },
	{ "FN2", "rightstick" }, { "UP", "dpup" }, { "DOWN", "dpdown" }, { "LEFT", "dpleft" }, { "RIGHT", "dpright" },
};
// the PlayStation's controls: their keys, input.ini's names and their inputs with nothing declared
struct Control
{
	const char *key;
	const char *ini;
	const char *inputs;
};
const Control Controls[] {
	{ "dpad_up", "up", "UP" }, { "dpad_down", "down", "DOWN" }, { "dpad_left", "left", "LEFT" }, { "dpad_right", "right", "RIGHT" },
	{ "cross", "cross", "B" }, { "circle", "circle", "A" }, { "square", "square", "Y" }, { "triangle", "triangle", "X" },
	{ "l1", "l1", "L1" }, { "r1", "r1", "R1" }, { "l2", "l2", "L2" }, { "r2", "r2", "R2" },
	{ "select", "select", "SELECT" }, { "start", "start", "START" },
};

// the arguments' (read before this file's other globals are made)
std::string& discPath()
{
	static std::string path;
	return path;
}
std::map<std::string, std::string>& controls()	// a control's inputs, where Deadstation says
{
	static std::map<std::string, std::string> list;
	return list;
}
std::string dataDir;
bool inputChanged = false;

int in = -1, out = -1;
std::string pending;
bool ready = false;
bool pauseWanted = false, resumeWanted = false;
bool hiding = false;

bool (*liveSetting)(const char *key, const char *value) = nullptr;

// the arguments, as early as anything can read the settings they make (some of the runtime's are read before main)
__attribute__((constructor(101))) void arguments(int argc, char **argv, char **)
{
	if (deadstation_module_start != nullptr)
		deadstation_module_start();
	for (int i = 1; i < argc; i++)
	{
		if (std::strcmp(argv[i], "--set") == 0 && i + 1 < argc)
		{
			const std::string pair = argv[++i];
			const size_t eq = pair.find('=');
			if (eq == std::string::npos)
				continue;
			const std::string key = pair.substr(0, eq), value = pair.substr(eq + 1);
			if (key.rfind(ControlPrefix, 0) == 0)
				controls()[key.substr(std::strlen(ControlPrefix))] = value;
			else if (deadstation_module_setting != nullptr)
				deadstation_module_setting(key.c_str(), value.c_str());
		}
		else if (discPath().empty())
			discPath() = argv[i];
	}
}

std::string field(const fs::path& manifest, const std::string& key)
{
	std::ifstream file(manifest);
	for (std::string line; std::getline(file, line); )
	{
		const size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		const auto trim = [](std::string s) {
			const size_t a = s.find_first_not_of(" \t\r");
			return a == std::string::npos ? std::string() : s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
		};
		if (trim(line.substr(0, eq)) == key)
			return trim(line.substr(eq + 1));
	}
	return "";
}

// input.ini, from the controls as they are
void writeInput()
{
	std::ofstream file("input.ini", std::ios::trunc);
	file << "[controller]\nenabled = true\ndevice = 0\ndeadzone = 3277\n\n[mapping]\n";
	for (const Control& control : Controls)
	{
		const auto chosen = controls().find(control.key);
		const std::string inputs = chosen != controls().end() ? chosen->second : control.inputs;
		std::string sources;
		size_t start = 0;
		while (start <= inputs.size())
		{
			const size_t end = std::min(inputs.find(',', start), inputs.size());
			const auto source = Sources.find(inputs.substr(start, end - start));
			if (source != Sources.end())
				sources += (sources.empty() ? "" : ",") + source->second;
			start = end + 1;
		}
		file << control.ini << " = " << sources << "\n";
	}
	file << "l3 =\nr3 =\nls_up = lefty-\nls_down = lefty+\nls_left = leftx-\nls_right = leftx+\n"
			"rs_up = righty-\nrs_down = righty+\nrs_left = rightx-\nrs_right = rightx+\n";
}

// the runtime's folder: this module's data. What the runtime wants beside it (its BIOS, its mods' packages) and the GPU's list
// of pipelines come with the module, copied when it is a new version; its Fast Loading and CD Speed on from the start.
void prepareFolder(const fs::path& moduleDir, const std::string& version)
{
	std::error_code ec;
	fs::create_directories(dataDir + "/saves", ec);
	if (chdir(dataDir.c_str()) != 0)
	{
		DS_LOG("recomp module: cannot use %s: %s", dataDir.c_str(), std::strerror(errno));
		return;
	}
	std::string kept;
	std::getline(std::ifstream(".module-version") >> std::ws, kept);
	if (kept != version)
	{
		fs::remove_all("mods/bundled", ec);
		fs::create_directories("mods", ec);
		fs::copy(moduleDir / "mods" / "bundled", "mods/bundled", fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
		if (ec)
			DS_LOG("recomp module: its mods not copied: %s", ec.message().c_str());
		ec.clear();
		fs::remove_all("bios", ec);
		fs::copy(moduleDir / "bios", "bios", fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
		if (ec)
			DS_LOG("recomp module: its BIOS not copied: %s", ec.message().c_str());
		ec.clear();
		if (fs::exists(moduleDir / "pipelines.keys", ec))
			fs::copy_file(moduleDir / "pipelines.keys", "pipelines.keys", fs::copy_options::overwrite_existing, ec);
		std::ofstream(".module-version", std::ios::trunc) << version << "\n";
	}
	std::string state;
	{
		std::ifstream file("mods/state.toml");
		std::getline(file, state, '\0');
	}
	if (state.find("psx.enhancement.fast-loading") == std::string::npos)
		std::ofstream("mods/state.toml", std::ios::trunc) << "format_version = 2\n\n"
				"[[package]]\nid = \"psx.enhancement.fast-loading\"\n\n[[package]]\nid = \"psx.enhancement.cd-speed\"\n\n"
				"[[feature]]\npackage_id = \"psx.enhancement.fast-loading\"\nid = \"fast-loading\"\nenabled = true\n[feature.values]\nuncapped = \"true\"\n\n"
				"[[feature]]\npackage_id = \"psx.enhancement.cd-speed\"\nid = \"cd-speed\"\nenabled = true\n[feature.values]\nmultiplier = \"8\"\n";
	writeInput();
}

void say(const char *line)
{
	if (out < 0)
		return;
	const std::string text = std::string(line) + "\n";
	if (write(out, text.data(), text.size()) != (ssize_t)text.size())
		DS_LOG("host link: Deadstation did not take \"%s\"", line);
}

void quit()
{
	DS_LOG("recomp module: quitting, as told");
	hostStop(0);
	_exit(0);
}

void set(const std::string& key, const std::string& value)
{
	if (key == "fps")
		hostShowFps(value == "on");
	else if (key == "label")
		hostCounterCells(value);
	else if (key.rfind(ControlPrefix, 0) == 0)
	{
		controls()[key.substr(std::strlen(ControlPrefix))] = value;
		inputChanged = true;
	}
	else if (liveSetting == nullptr || !liveSetting(key.c_str(), value.c_str()))
		DS_LOG("host link: %s = %s applies at the next start", key.c_str(), value.c_str());
}

void command(const std::string& line)
{
	if (line == "pause")
		pauseWanted = true;
	else if (line == "resume")
		resumeWanted = true;
	else if (line == "quit")
		quit();
	else if (line.rfind("set ", 0) == 0)
	{
		const size_t space = line.find(' ', 4);
		if (space != std::string::npos)
			set(line.substr(4, space - 4), line.substr(space + 1));
		else
			DS_LOG("host link: bad command: %s", line.c_str());
	}
	else
		DS_LOG("host link: unknown command: %s", line.c_str());
}

void commands()
{
	char data[512];
	for (;;)
	{
		const ssize_t size = read(in, data, sizeof(data));
		if (size == 0)
		{
			DS_LOG("host link: Deadstation has gone");
			quit();
		}
		if (size < 0)
			return;
		pending.append(data, (size_t)size);
		for (size_t end; (end = pending.find('\n')) != std::string::npos; pending.erase(0, end + 1))
			command(pending.substr(0, end));
	}
}

// fb0 in NextUI's own mode, with the frame on the screen scaled up to it: the menu is drawn at the panel's size, over that
bool toPanelSize(Video& video)
{
	if (video.panel().nativeMode())
		return false;
	std::vector<uint8_t> rgb;
	int w = 0, h = 0, width = 0, height = 0;
	if (!video.panel().readShown(rgb, w, h) || !video.panel().beginPictures(width, height))
		return false;
	std::vector<uint32_t> bgra((size_t)width * height);
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			const uint8_t *p = &rgb[((size_t)(y * h / height) * w + x * w / width) * 3];
			bgra[(size_t)y * width + x] = 0xff000000u | (uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2];
		}
	video.panel().showPicture(bgra.data());
	return true;
}

// until resume: the guest held, the screen Deadstation's
void hold()
{
	pauseWanted = hiding = false;
	psx_gpu_idle();
	Video& video = hostVideo();
	video.panel().drain();
	const bool pictures = toPanelSize(video);
	psx_host_audio_hold(1);
	say("paused");
	DS_LOG("recomp module: paused");
	while (!resumeWanted)
	{
		pollfd f = { in, POLLIN, 0 };
		if (poll(&f, 1, -1) > 0)
			commands();
	}
	resumeWanted = false;
	if (pictures)
		video.panel().endPictures(Panel::After::LastFrame);
	if (inputChanged)
	{
		inputChanged = false;
		writeInput();
		psx_host_reload_input();
	}
	psx_host_audio_hold(0);
	hostMenuClosed();
	DS_LOG("recomp module: resumed");
}

}	// namespace

extern "C" void deadstation_live_settings(bool (*apply)(const char *key, const char *value))
{
	liveSetting = apply;
}

void hostLinkPoll()
{
	if (in < 0)
		return;
	if (!ready)
	{
		ready = true;
		say("ready");
	}
	commands();
	if (hiding || (pauseWanted && !hostHideFps()))
		hold();
	else if (pauseWanted)
	{
		// the counter off the picture the menu is drawn over: one more frame without it
		pauseWanted = false;
		hiding = true;
	}
}

extern "C" int __wrap_main(int, char **argv)
{
	if (const char *fds = std::getenv("DEADUI_HOST"); fds == nullptr || std::sscanf(fds, "%d,%d", &in, &out) != 2)
	{
		in = out = -1;
		DS_LOG("host link: no host (DEADUI_HOST): the game plays with no menu");
	}
	else
	{
		fcntl(in, F_SETFL, O_NONBLOCK);
		fcntl(in, F_SETFD, FD_CLOEXEC);
		fcntl(out, F_SETFD, FD_CLOEXEC);
	}
	std::error_code ec;
	const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
	if (ec)
	{
		DS_LOG("recomp module: where is its executable? %s", ec.message().c_str());
		return 1;
	}
	const fs::path moduleDir = exe.parent_path();
	const fs::path manifest = moduleDir / "module.cfg";
	const char *home = std::getenv("DEADSTATION_HOME");
	dataDir = home != nullptr ? std::string(home) + "/recomp-data/" + field(manifest, "id") : moduleDir.string();
	prepareFolder(moduleDir, field(manifest, "version"));
	// its folder as the runtime finds it: argv[0]'s
	std::vector<std::string> args { dataDir + "/" + exe.filename().string(), "--no-launcher", "--renderer", "vulkan",
		"--game", (moduleDir / "game.toml").string(), "--memcard-dir", dataDir + "/saves" };
	if (!discPath().empty())
	{
		args.push_back("--disc");
		args.push_back(discPath());
	}
	std::vector<char *> made;
	for (std::string& arg : args)
		made.push_back(arg.data());
	made.push_back(nullptr);
	(void)argv;
	return __real_main((int)made.size() - 1, made.data());
}
