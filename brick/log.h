// Deadstation's log: stderr, which launch.sh sends to NextUI's log folder.
#pragma once
#include <cstdio>

#define DS_LOG(...) do { std::fprintf(stderr, "Deadstation: " __VA_ARGS__); std::fputc('\n', stderr); } while (0)
