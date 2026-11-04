# Deadstation recomp SDK

The runtime and port layer for Deadstation's recomp modules. A recomp module is a recompiled build of one PlayStation game for the
TrimUI Brick: the game's MIPS code translated to C at build time from the builder's own disc, and played under Deadstation in place of
its emulation. A module is its own repository, with this one as a submodule (`sdk/`), released as a tarball.

The translator and runtime are [PSXRecomp](https://github.com/RetroPortingToolKit/psxrecomp), a submodule here at an audited pin, brought to
this SDK's patches by `tools/prepare-psxrecomp.sh`.

**Noncommercial only.** PSXRecomp is PolyForm Noncommercial 1.0.0, so every module's binary is too. This SDK and its modules use the same licence (`LICENSE`).

## What every module follows

1. **A module, not a replacement.** Deadstation stays running and hosts the module as a child process. MENU, the menu, Options and
   the Recomp Store are Deadstation's. The module is the game: it draws, plays sound, and pauses, resumes and quits when told to,
   through the port layer (`brick/host_link.cpp`).
2. **No game data in git.** No disc images, nothing extracted from them, no translated code and no overlay captures.
3. **The disc is checked.** The manifest lists every disc the module plays, by the SHA-1 its CHD's header records. Deadstation offers the
   module only for those discs.
4. **Whatever a module lets the player choose is a setting declared in its manifest,** shown in Options.

## What's generic here, and what's the game's

This SDK helps make a recomp. It holds nothing of any one game: a game's lights, moods, menus and fixes are its module's, in its plugin. Where a module needs to draw in its own way, it brings its own
shaders and code, and the SDK gives it the place to put them:

- `deadstation_module_layer` (`brick/video.h`): a `VideoLayer` of its own, drawn with each frame over what the game drew.
- `deadstation_world_shaders`, `deadstation_world_numbers` (`brick/replay.h`): its own shaders for a world mesh, and numbers for them each frame.
- `deadstation_module_shaders`, `deadstation_module_sources` (`brick/brick.cmake`): how its CMakeLists.txt builds them.

## Pace and sound

A module runs at the emulator's pace: the panel at 789 lines a frame, about 59.85 Hz, close to a real PlayStation, while it runs (`Panel::matchRate`). Sound isn't tied to that. It's mixed on the audio device's thread, exactly what the speaker asks for, with nothing resampled or stretched (PSXRecomp patch 0033); the game's thread only sets the sound chip's state. An audio track is read a second ahead of the drive's head as the game sees it. A module with its own CD feed steers by `spu_cd_pace()` and holds about 400 ms.

## A module's repository

```
module.cfg          the manifest
store.json          what its page in the Recomp Store says
README.md           short: the game, Supported Formats, Features, Files, Building, Credits
game.toml           the game: its disc, boot file, runtime settings (PSXRecomp)
seeds/              the functions the translation starts from
CMakeLists.txt      the module's build, on psxrecomp_add_game_runtime and brick/brick.cmake
src/plugin.cpp      native code replacing the game's libraries, and the module's hooks (game.toml's mod_function_entry_funcs)
mods/preloaded/     the plugin's package, which turns it on
pipelines.keys      optional: the GPU pipelines it draws, built ahead of use
sdk/                this repository, a submodule
```

Ignore `build-brick/`, `disc/`, `generated/`, `overlays/` and every disc format.

### module.cfg

```
id = mygame               unique; the folder it installs in
name = My Game
title = My Game           what the menu calls it
version = 1.0.0
exec = My_Game_recomp_module      the executable, beside the manifest
needs = 1.0.0             the oldest Deadstation it runs under: the store offers it to no older one
disc = 0123abcd...        each disc it plays: the SHA-1 its CHD's header records (one line each)
setting = scale | Resolution | 1=Original, 2=1024 x 768 | 2 | What its row says when selected
                          each setting: key | name | value=label, ... | default | description, if any
```

Deadstation shows settings in Options, with Core on the module, keeps the values in the module's own `recomp-data/<id>/settings.cfg`, and
hands every value to the module as it starts, as `--set <key>=<value>`. **Show FPS** is always the first row there, undeclared.

A module turns its settings into what its plugin reads with two optional functions (`brick/host_link.h`), as its process starts:
`deadstation_module_start()` and `deadstation_module_setting(key, value)`. One the plugin can apply while playing is registered with
`deadstation_live_settings` (`brick/psx_sdk.h`). Say in the description of any other that it takes effect the next time the game starts.

### Buttons

The port layer plays the game on this layout, with nothing declared:

| Control | Key | Default |
|---|---|---|
| D-pad | `dpad_up`, `dpad_down`, `dpad_left`, `dpad_right` | `UP`, `DOWN`, `LEFT`, `RIGHT` |
| Cross, Circle, Square, Triangle | `cross`, `circle`, `square`, `triangle` | `B`, `A`, `Y`, `X` |
| L1, R1, L2, R2 | `l1`, `r1`, `l2`, `r2` | `L1`, `R1`, `L2`, `R2` |
| Select, Start | `select`, `start` | `SELECT`, `START` |

To let the player remap them, declare each control they may change in `module.cfg`, with its default:

```
control = cross | Cross | B      key | name | the pad's inputs, comma separated, or NONE
```

Options then has a **Controls** page for the module, drawn by Deadstation: A waits for a press, X clears, and a press taken leaves any other
declared control that had it. The inputs are `A B X Y L1 R1 L2 R2 SELECT START FN1 FN2 UP DOWN LEFT RIGHT`, by the labels on the device. MENU
is Deadstation's and cannot be bound. A control not declared stays on its default, and a module that declares none has no page. The values reach
the module as `control.<key>`, and apply when the game goes on. They are the module's alone: nothing comes from the emulator's Controls.

### store.json

Every key is optional. The changelog is for the version in `module.cfg`, and `formats` lists the same discs:

```json
{
  "description": "The game in a line or two.",
  "formats": [{ "format": "CHD", "region": "USA", "sha1": "0123abcd..." }],
  "features": ["One line for each thing it adds."],
  "changelog": ["One line for each change in this version."],
  "repo": "https://github.com/..."
}
```

## How Deadstation runs a module

Installed modules are in `<DEADSTATION_HOME>/recomp/<id>/`. When a game's core is a module, Deadstation starts it as `<exec> <disc>` and sleeps
while it plays, so the emulator never starts. The port layer makes PSXRecomp's arguments, and runs it with `<DEADSTATION_HOME>/recomp-data/<id>/` as its
folder: memory cards, caches and everything else it writes go there, and nothing outside those two folders. Deleting a module removes both.

Any error of a module's ends its run: it cannot start, it crashes, it exits with anything but 0, or it does not answer `pause`. Deadstation shows the
error with one button, `A TRY EMULATOR CORE`. The game's core becomes the emulator and the game starts again in it. The player can switch back in Options,
and update or delete the module in the Recomp Store.

## Building a module

From the module's repository, with your own disc. Fetch PSXRecomp's submodule, never recursively: everything a build needs is at its pin, and its nested
submodules are not used. Every step runs without network (`tools/sbx`).

```
git submodule update --init sdk
git -C sdk submodule update --init psxrecomp
sdk/tools/toolchain/toolchain.sh
sdk/tools/build-module.sh /path/to/your/disc.chd
sdk/tools/pack-module.sh
```

`build-module.sh` makes `build-brick/<id>/`, what Deadstation installs. `pack-module.sh` makes the tarball for its release. The toolchain needs
`docker pull ghcr.io/loveretro/tg5040-toolchain:latest` first. On the host: clang, lld, gcc, CMake, Ninja, Python 3, bubblewrap, SDL3, `vulkan-headers` and `glslang`.

A game that loads code from its disc as it plays needs its overlays captured first: PSXRecomp's tools play the game and write `overlays/overlay_captures.json`,
which `build-module.sh` bakes in. That file holds the game's code: never in git.

## Getting listed

1. Open a pull request adding your repository to Deadstation's `store/registry.txt`.
2. Publish a GitHub release with the tarball attached. Tag it with the version in `module.cfg`, and raise that version for every release.

The store's list picks up your latest release the next time it is rebuilt: on a schedule of an hour, which GitHub does not guarantee, so it can take longer.

## Licences

PolyForm Noncommercial 1.0.0 (`LICENSE`); `THIRD_PARTY_NOTICES.md` lists what it builds on.
