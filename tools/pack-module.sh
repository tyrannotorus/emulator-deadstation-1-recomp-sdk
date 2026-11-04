#!/bin/sh
# Pack a built module (build-module.sh) as build-brick/release/recomp-<id>-<version>.tar.gz, the file to attach to
# its release. The same build packs to the same bytes. store.json's formats must be the manifest's discs.
#   sdk/tools/pack-module.sh
set -e
[ -f module.cfg ] || { echo "pack-module: no module.cfg in this folder"; exit 2; }
M=$(sed -n 's/^id *= *//p' module.cfg)
[ -f "build-brick/$M/module.cfg" ] || { echo "pack-module: no build-brick/$M (build-module.sh)"; exit 2; }
mkdir -p build-brick/release
python3 -I - "build-brick/$M" build-brick/release <<'EOF'
import gzip, hashlib, json, os, sys, tarfile

src, out = sys.argv[1:3]
cfg, discs = {}, []
for line in open(os.path.join(src, "module.cfg"), encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = (s.strip() for s in line.split("=", 1))
    if k == "disc":
        discs.append(v)
    else:
        cfg.setdefault(k, v)
store = json.load(open(os.path.join(src, "store.json"), encoding="utf-8"))
if sorted(d.get("sha1", "") for d in store.get("formats", [])) != sorted(discs):
    sys.exit("pack-module: store.json's formats must list the manifest's discs")
path = os.path.join(out, f"recomp-{cfg['id']}-{cfg['version']}.tar.gz")

def normal(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() or os.access(os.path.join(src, info.name), os.X_OK) else 0o644
    return info

with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz, \
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
    for root, dirs, files in os.walk(src):
        dirs.sort()
        rel = os.path.relpath(root, src)
        if rel != ".":
            tar.add(root, arcname=rel, recursive=False, filter=normal)
        for f in sorted(files):
            tar.add(os.path.join(root, f), arcname=os.path.normpath(os.path.join(rel, f)), filter=normal)
data = open(path, "rb").read()
print(f"pack-module: {path} ({len(data)} bytes, SHA-256 {hashlib.sha256(data).hexdigest()})")
EOF
