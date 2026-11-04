// A module's own part in how it starts (host_link.cpp): turning its settings, from module.cfg by way of Deadstation, into
// what its plugin and the runtime read. Both optional.
#pragma once

// as the module's process starts, before anything else of it
extern "C" __attribute__((weak)) void deadstation_module_start(void);
// each setting at its value, then
extern "C" __attribute__((weak)) void deadstation_module_setting(const char *key, const char *value);

// after each frame, on the module's thread: Deadstation's commands, and the guest held when it says pause
void hostLinkPoll();
