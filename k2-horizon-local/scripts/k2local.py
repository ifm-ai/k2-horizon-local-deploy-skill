#!/usr/bin/env python3
"""Run K2 Horizon (0.9B, 3.7B or 7B) on your own computer.

First time, in this order:
  k2local.py check                     what fits on this computer
  k2local.py install                   download llama.cpp (--build latest for the newest)
  k2local.py download --size 3.7B      download and verify a model
  k2local.py serve --size 3.7B         start it in the background
  k2local.py chat                      chat in the terminal (--window for a new window)
  k2local.py ask "question"            ask one question

Or use Ollama instead of install / download / serve:
  k2local.py ollama --size 3.7B        chat through Ollama
  k2local.py opencode --size 7B        code with OpenCode, on top of Ollama

Other commands: status, stop.
Files are kept in ~/k2-horizon-local (change it with --dir or K2_DIR).
Needs Python 3.8+, nothing else.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

TESTED_BUILD = "b11619"  # llama.cpp build tested with K2 Horizon
RELEASES = "https://github.com/ggml-org/llama.cpp/releases"
PORT = 8080
CONTEXT = 16384
QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"]
EFFORTS = ["low", "medium", "high"]
HEADERS = {"User-Agent": "k2-horizon-local"}

OLLAMA_REPO = "ifm/k2-horizon"
OLLAMA_MIN = (0, 40, 4)

# IFM's GGUF repos. IFM names the 0.9B files "1B" and the 3.7B files "4B".
# File sizes are in GB. Speeds were measured on an M4 Pro with 24 GB.
MODELS = {
    "0.9B": {
        "repo": "IFM/K2-Horizon-0.9B-GGUF",
        "prefix": "K2-Horizon-1B",
        "temp": 0.6,
        "gb": {"Q4_K_M": 0.67, "Q5_K_M": 0.77, "Q6_K": 0.89, "Q8_0": 1.15, "BF16": 2.16},
        "speed": "~165 tokens/s (Q8)",
        "made_for": "watches, glasses, quick commands",
        "good": "instant replies, reminders, simple tool calls, short translations",
        "weak": "makes reasoning mistakes, e.g. split 86 three ways as 28+28+30",
    },
    "3.7B": {
        "repo": "IFM/K2-Horizon-3.7B-GGUF",
        "prefix": "K2-Horizon-4B",
        "temp": 1.0,
        "gb": {"Q4_K_M": 3.16, "Q5_K_M": 3.64, "Q6_K": 4.16, "Q8_0": 5.39, "BF16": 10.13},
        "speed": "~46 tokens/s (Q8)",
        "made_for": "phones and everyday laptop use",
        "good": "summaries, document questions, tool calls, code fixes",
        "weak": "Arabic phrasing and transliteration were partly wrong",
    },
    "7B": {
        "repo": "IFM/K2-Horizon-7B-GGUF",
        "prefix": "K2-Horizon-7B",
        "temp": 1.0,
        "gb": {"Q4_K_M": 5.59, "Q5_K_M": 6.47, "Q6_K": 7.39, "Q8_0": 9.57, "BF16": 18.01},
        "speed": "~40 tokens/s (Q4)",
        "made_for": "laptops and developer workstations",
        "good": "coding, agents, reviews, research",
        "weak": "slowest of the three; needs 16 GB+ memory",
    },
}

FORMATS = {
    "gguf": (
        "GGUF, compressed",
        "A compressed single file run by llama.cpp: the smallest download and the fastest "
        "replies, with a small loss in quality.",
    ),
    "gguf-bf16": (
        "GGUF, full quality (BF16)",
        "The full-precision weights in one file for llama.cpp: original quality, but about "
        "twice the size and ~30% slower.",
    ),
}

ARGS = None


# Storage and state

def home():
    path = os.path.expanduser(ARGS.dir or os.environ.get("K2_DIR") or "~/k2-horizon-local")
    os.makedirs(path, exist_ok=True)
    return path


def load_state():
    try:
        with open(os.path.join(home(), "state.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(**changes):
    state = load_state()
    state.update(changes)
    with open(os.path.join(home(), "state.json"), "w") as f:
        json.dump(state, f, indent=1)


def model_key(size, fmt):
    """How a download is stored in the state file: "7B" or "7B-bf16"."""
    return size + "-bf16" if fmt == "gguf-bf16" else size


def key_size(key):
    return key.split("-")[0]


def key_format(key):
    return "gguf-bf16" if key.endswith("-bf16") else "gguf"


def label(key):
    return key_size(key) + (" BF16" if key_format(key) == "gguf-bf16" else "")


def downloaded_models(state):
    # Older versions of the script could also record "-original" models; skip those.
    models = state.get("models") or {}
    return {k: v for k, v in models.items() if not k.endswith("-original")}


def command_hint(python=None):
    """The command to type later, with --dir and --port when they aren't the defaults."""
    python = python or ("py" if platform.system() == "Windows" else "python3")
    hint = f'{python} "{os.path.abspath(__file__)}"'
    if home() != os.path.expanduser("~/k2-horizon-local"):
        hint += f' --dir "{home()}"'
    if ARGS.port != PORT:
        hint += f" --port {ARGS.port}"
    return hint


# Hardware

def total_ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 2**20
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                 "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**30
    except Exception:
        pass
    return None


def gpu_kind():
    if platform.system() == "Darwin":
        return "apple-metal" if platform.machine() == "arm64" else "none"
    if shutil.which("nvidia-smi"):
        try:
            subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=10)
            return "nvidia"
        except Exception:
            pass
    if shutil.which("vulkaninfo"):
        return "vulkan"
    return "none"


def keep_awake():
    """Downloads stall when the computer goes to sleep."""
    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    elif platform.system() == "Windows":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def fit(gb, ram):
    # Keep the file under a quarter of the memory. On a 24 GB Mac the 9.6 GB file
    # swapped to disk and dropped to ~3 tokens/s, while 5.6 GB ran at ~40.
    if ram is None:
        return "unknown"
    if gb <= ram * 0.25:
        return "comfortable"
    if gb <= ram * 0.4:
        return "tight"
    return "too big"


def best_quant(size, ram):
    """The best quality file that fits comfortably, or the smallest one."""
    for quant in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"):
        if fit(MODELS[size]["gb"][quant], ram) == "comfortable":
            return quant
    return "Q4_K_M"


def pick_quant(size):
    if ARGS.format == "gguf-bf16" and ARGS.quant not in (None, "BF16"):
        sys.exit("--format gguf-bf16 is the BF16 file, so leave out --quant.")
    if ARGS.format == "gguf-bf16" or ARGS.quant == "BF16":
        return "BF16"
    return ARGS.quant or best_quant(size, total_ram_gb())


# check

def cmd_check():
    ram = total_ram_gb()
    free = shutil.disk_usage(home()).free / 1e9
    gpu = gpu_kind()
    models = downloaded_models(load_state())

    sizes = []
    for size, m in MODELS.items():
        quant = best_quant(size, ram)
        done = models.get(size, {})
        sizes.append({
            "size": size,
            "quant": quant,
            "file_gb": m["gb"][quant],
            "fit": fit(m["gb"][quant], ram),
            "speed_measured": m["speed"],
            "made_for": m["made_for"],
            "good_at": m["good"],
            "watch_out": m["weak"],
            "already_downloaded": done.get("quant") if done.get("verified") and os.path.exists(done.get("path", "")) else None,
        })

    formats = []
    for fmt, (title, about) in FORMATS.items():
        fits = []
        for size, m in MODELS.items():
            gb = m["gb"]["BF16"] if fmt == "gguf-bf16" else m["gb"][best_quant(size, ram)]
            done = bool(models.get(model_key(size, fmt), {}).get("verified"))
            fits.append({"size": size, "gb": gb, "fit": fit(gb, ram), "downloaded": done})
        formats.append({"format": fmt, "title": title, "about": about, "sizes": fits})

    comfortable = [s["size"] for s in sizes if s["fit"] == "comfortable"]
    if comfortable:
        recommended = comfortable[-1]  # the biggest one that fits comfortably
    elif sizes[0]["fit"] != "too big":
        recommended = "0.9B"
    else:
        recommended = None

    info = {
        "os": platform.system(),
        "arch": platform.machine().lower(),
        "ram_gb": round(ram, 1) if ram else None,
        "free_disk_gb": round(free, 1),
        "gpu": gpu,
        "options": sizes,
        "recommended_size": recommended,
        "formats": formats,
        "default_format": "gguf",
        "dir": home(),
    }
    print(json.dumps(info, indent=1, ensure_ascii=False))
    print()

    memory = f"Memory: {ram:.0f} GB, " if ram else ""
    print(f"{memory}free disk: {free:.0f} GB, GPU: {gpu}")
    marks = {"comfortable": "fits comfortably", "tight": "fits, close other apps",
             "too big": "too big", "unknown": "unknown"}
    # Smallest to largest, so "the middle one" or "2" is clear.
    for n, s in enumerate(sizes, 1):
        done = f"  [downloaded: {s['already_downloaded']}]" if s["already_downloaded"] else ""
        print(f"  {n}. {s['size']:>5}  {s['quant']:<7} {s['file_gb']:>5.2f} GB  "
              f"{marks[s['fit']]:<24} {s['speed_measured']}{done}")
    print(f"\nRecommended: {recommended}" if recommended else "\nEven 0.9B is a tight fit here.")

    print("\nFormats (the list above uses a, the default):")
    for letter, f in zip("ab", formats):
        print(f"  {letter}. {f['title']}: {f['about']}")
        if f["format"] == "gguf-bf16":
            print("     " + ", ".join(
                f"{s['size']} {s['gb']:.1f} GB {marks[s['fit']]}" + (" [downloaded]" if s["downloaded"] else "")
                for s in f["sizes"]))

    for s in sizes:
        if s["fit"] != "too big" and free < s["file_gb"] + 1:
            print(f"Not enough free disk for {s['size']} ({s['file_gb']:.1f} GB).")
    if gpu == "none" and platform.system() != "Darwin":
        print("No GPU found, so the model runs on the CPU. It works, but several times slower.")
    save_state(check=info)


# install

def latest_build():
    # Follow the /releases/latest redirect instead of using the GitHub API: the API allows
    # only 60 calls an hour per network, which a shared office network runs out of.
    try:
        req = urllib.request.Request(f"{RELEASES}/latest", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            tag = r.geturl().rstrip("/").rsplit("/", 1)[-1]
        if not re.fullmatch(r"b\d+", tag):
            # Stable releases are tagged v0.x and name their build in nightly-tag.txt.
            req = urllib.request.Request(f"{RELEASES}/download/{tag}/nightly-tag.txt", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as r:
                tag = r.read().decode().strip()
    except Exception:
        return None
    return tag if re.fullmatch(r"b\d+", tag) else None


def archive_names(build, backend):
    system = platform.system()
    arm = platform.machine().lower() in ("arm64", "aarch64")
    b = build
    if system == "Darwin":
        return [f"llama-{b}-bin-macos-{'arm64' if arm else 'x64'}.tar.gz"]
    if system == "Linux":
        if arm:
            return [f"llama-{b}-bin-ubuntu-{'vulkan-arm64' if backend == 'vulkan' else 'arm64'}.tar.gz"]
        if backend == "cuda":
            return [f"llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz"]
        if backend == "vulkan":
            return [f"llama-{b}-bin-ubuntu-vulkan-x64.tar.gz"]
        return [f"llama-{b}-bin-ubuntu-x64.tar.gz"]
    if system == "Windows":
        if arm:
            return [f"llama-{b}-bin-win-{'vulkan' if backend == 'vulkan' else 'cpu'}-arm64.zip"]
        if backend == "cuda":
            return [f"llama-{b}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
        if backend == "vulkan":
            return [f"llama-{b}-bin-win-vulkan-x64.zip"]
        return [f"llama-{b}-bin-win-cpu-x64.zip"]
    sys.exit(f"Unsupported OS: {system}")


def download_archives(build, backend):
    """Returns the downloaded paths, or None if the release doesn't have these files."""
    paths = []
    for name in archive_names(build, backend):
        url = f"{RELEASES}/download/{build}/{name}"
        dest = os.path.join(home(), name)
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as e:
            for p in paths:
                os.remove(p)
            if e.code == 404:
                return None
            sys.exit(f"Couldn't download {url}: {e}")
        except OSError as e:
            sys.exit(f"Couldn't download {url}: {e}. Check the internet connection and run install again.")
        paths.append(dest)
    return paths


def find_server(root):
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for folder, _, files in os.walk(root):
        if exe in files:
            return os.path.join(folder, exe)
    return None


def cmd_install():
    if platform.system() == "Darwin":
        backend = "metal" if platform.machine() == "arm64" else "cpu"  # the macOS builds include Metal
    elif ARGS.backend == "auto":
        backend = {"nvidia": "cuda", "vulkan": "vulkan"}.get(gpu_kind(), "cpu")
    else:
        backend = ARGS.backend

    root = os.path.join(home(), "llama")
    state = load_state()
    installed = find_server(root)
    if installed and not ARGS.force and not ARGS.build:
        print(f"llama.cpp {state.get('build') or ''} is already installed: {installed}")
        return

    if ARGS.build in (None, "tested"):
        build = TESTED_BUILD
    elif ARGS.build == "latest":
        build = latest_build()
        if not build:
            sys.exit(f"Couldn't look up the latest llama.cpp release at {RELEASES}/latest. "
                     "Check the internet connection and run install again.")
    else:
        build = ARGS.build

    if installed and build == state.get("build") and not ARGS.force:
        print(f"llama.cpp {build} is already installed: {installed}")
        return
    if installed and pid_alive(state.get("pid")):
        sys.exit("The model is running from the installed llama.cpp. Run stop first, then install again.")

    archives = download_archives(build, backend)
    if not archives:
        sys.exit(f"llama.cpp {build} has no {' + '.join(archive_names(build, backend))}. "
                 f"See {RELEASES} or try --backend cpu.")

    # Start clean, so an old llama-server isn't left next to the new one.
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for path in archives:
        if path.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                z.extractall(root)
        else:
            with tarfile.open(path) as t:
                t.extractall(root)
        os.remove(path)

    server = find_server(root)
    if not server:
        sys.exit(f"Unpacked llama.cpp but couldn't find llama-server. Look inside {root}")
    if platform.system() != "Windows":
        for folder, _, files in os.walk(os.path.dirname(server)):
            for name in files:
                path = os.path.join(folder, name)
                os.chmod(path, os.stat(path).st_mode | 0o111)
    if platform.system() == "Darwin":
        # Otherwise macOS refuses to run the downloaded binaries.
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", root], stderr=subprocess.DEVNULL)

    out = subprocess.run([server, "--version"], capture_output=True, text=True)
    version = (out.stdout + out.stderr).strip()
    print(version.splitlines()[0] if version else "(no version output)")
    save_state(server=server, backend=backend, build=build)
    if build == TESTED_BUILD:
        print(f"Installed llama.cpp {build} ({backend}), the build tested with K2 Horizon.")
    else:
        print(f"Installed llama.cpp {build} ({backend}). It hasn't been tested with K2 Horizon; "
              "if anything misbehaves, go back with: install --build tested")


# download

def fetch_file(url, dest, total, sha256):
    """Download in 8 parallel parts (a single connection kept stalling), resuming
    whatever is already there, then check the size and checksum."""
    keep_awake()
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > total:
        os.remove(dest)
        have = 0
    name = os.path.basename(dest)
    print(f"Downloading {name}: {total / 1e9:.2f} GB ({have / 1e9:.2f} GB already here)")

    parts = 8
    left = total - have
    ranges = []
    if left:
        ranges = [(have + left * i // parts, have + left * (i + 1) // parts - 1) for i in range(parts)]
    progress = [0] * len(ranges)
    failed = []

    def download_part(i, start, end):
        part = f"{dest}.part{i}"
        for attempt in range(40):
            done = os.path.getsize(part) if os.path.exists(part) else 0
            if start + done > end:
                return
            try:
                req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={start + done}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        f.write(chunk)
                        progress[i] = f.tell()
            except Exception:
                time.sleep(min(30, 2 + attempt * 2))
        failed.append(i)

    threads = [threading.Thread(target=download_part, args=(i, start, end), daemon=True)
               for i, (start, end) in enumerate(ranges)]
    for t in threads:
        t.start()
    last, last_time = 0, time.time()
    while any(t.is_alive() for t in threads):
        for _ in range(15):  # report every 15 seconds
            time.sleep(1)
            if not any(t.is_alive() for t in threads):
                break
        now = sum(progress)
        speed = (now - last) / (time.time() - last_time) / 1e6
        print(f"  {(have + now) / 1e9:5.2f} / {total / 1e9:.2f} GB, {speed:5.1f} MB/s")
        last, last_time = now, time.time()
    if failed:
        sys.exit("Some parts failed after many retries. Run the same command again to resume.")

    with open(dest, "ab") as out:
        for i in range(len(ranges)):
            part = f"{dest}.part{i}"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
            os.remove(part)

    print("Checking the checksum against Hugging Face ...")
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if os.path.getsize(dest) != total or h.hexdigest() != sha256:
        os.rename(dest, dest + ".bad")
        sys.exit(f"Checksum mismatch in {name}. The damaged file was renamed to .bad. Run download again.")


def cmd_download():
    size = ARGS.size
    if not size:
        sys.exit("Choose a size: download --size 0.9B | 3.7B | 7B   (run check to see what fits)")
    quant = pick_quant(size)
    fmt = "gguf-bf16" if quant == "BF16" else "gguf"
    key = model_key(size, fmt)
    repo = MODELS[size]["repo"]
    name = f"{MODELS[size]['prefix']}-{quant}.gguf"
    dest = os.path.join(home(), name)
    note = {"0.9B": ' (IFM names the 0.9B file "1B")', "3.7B": ' (IFM names the 3.7B file "4B")'}.get(size, "")

    models = downloaded_models(load_state())
    prev = models.get(key, {})
    if prev.get("verified") == name and os.path.exists(dest) and os.path.getsize(dest) == prev.get("verified_size"):
        print(f"{label(key)} is already downloaded and verified: {dest}{note}")
    else:
        with urllib.request.urlopen(urllib.request.Request(
                f"https://huggingface.co/api/models/{repo}/tree/main", headers=HEADERS), timeout=60) as r:
            files = {f["path"]: f for f in json.loads(r.read())}
        if name not in files:
            sys.exit(f"{name} isn't in {repo} on Hugging Face.")
        total, sha256 = files[name]["lfs"]["size"], files[name]["lfs"]["oid"]
        fetch_file(f"https://huggingface.co/{repo}/resolve/main/{name}", dest, total, sha256)
        prev = {"verified": name, "verified_size": total}
        flag = " --format gguf-bf16" if fmt == "gguf-bf16" else ""
        print(f"Verified {name} (sha256 matches Hugging Face).{note} Start it with: serve --size {size}{flag}")

    state = load_state()
    all_models = state.get("models") or {}
    all_models[key] = {**prev, "path": dest, "quant": quant}
    save_state(models=all_models, active=key)


# serve, stop, status

def pid_alive(pid):
    if not pid:
        return False
    if platform.system() == "Windows":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def port_in_use():
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", ARGS.port)) == 0


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ARGS.port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def background_flags():
    if platform.system() == "Windows":
        DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP = 0x00000008, 0x00000200
        return {"creationflags": DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def pick_model(state):
    """--format if given, otherwise the last one used for that size, then compressed, then BF16."""
    models = downloaded_models(state)
    active = state.get("active") if state.get("active") in models else None
    size = ARGS.size or (key_size(active) if active else None)
    if not size:
        return None
    if ARGS.format:
        return model_key(size, ARGS.format)
    if active and key_size(active) == size:
        return active
    return size if size in models or f"{size}-bf16" not in models else f"{size}-bf16"


def print_usage_hints():
    hint = command_hint()
    print(f"  Web chat: http://127.0.0.1:{ARGS.port}  (open it in a browser)")
    print(f"  Chat: {hint} chat")
    print(f'  Ask:  {hint} ask "your question"')
    print(f"  Stop: {hint} stop")


def cmd_serve():
    state = load_state()
    models = downloaded_models(state)
    key = pick_model(state)
    if not key or key not in models or not os.path.exists(models[key].get("path", "")):
        have = ", ".join(label(k) for k in models) or "none"
        size = key_size(key) if key else "3.7B"
        flag = " --format gguf-bf16" if key and key_format(key) == "gguf-bf16" else ""
        sys.exit(f"{label(key) if key else 'That size'} isn't downloaded yet (downloaded: {have}). "
                 f"Run: download --size {size}{flag}")
    server = state.get("server")
    if not server or not os.path.exists(server):
        sys.exit("llama.cpp isn't installed yet. Run: install")

    if port_in_use():
        if pid_alive(state.get("pid")) and healthy():
            if state.get("serving") == key:
                print(f"{label(key)} is already running at http://127.0.0.1:{ARGS.port}")
                print_usage_hints()
                return
            sys.exit(f"{label(state.get('serving') or '?')} is running. Run stop first, then serve again.")
        sys.exit(f"Port {ARGS.port} is used by another program. Stop it, or use another port: "
                 "--port 8081 (with ask and chat too).")

    # IFM's recommended settings, set on the server so the web chat and other apps get them too.
    # --jinja lets llama.cpp split the replies into reasoning, answer and tool calls.
    cmd = [
        server, "-m", models[key]["path"], "--host", "127.0.0.1", "--port", str(ARGS.port),
        "--jinja", "-c", str(ARGS.ctx), "-ngl", "99",
        "--temp", str(MODELS[key_size(key)]["temp"]), "--top-p", "0.95",
        "--chat-template-kwargs", json.dumps({"reasoning_effort": "high"}),
    ]
    log_path = os.path.join(home(), "server.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=os.path.dirname(server), **background_flags())
    save_state(pid=proc.pid, port=ARGS.port, serving=key, active=key)
    print(f"Starting K2-Horizon-{label(key)} (pid {proc.pid}) ...")

    start = time.time()
    while time.time() - start < 900:  # the first load of a big file can be slow
        if proc.poll() is not None:
            with open(log_path) as f:
                sys.exit("The server stopped while loading. Last log lines:\n" + f.read()[-1500:])
        if healthy():
            print(f"{label(key)} is ready after {time.time() - start:.0f}s at http://127.0.0.1:{ARGS.port} "
                  "(OpenAI-compatible API at /v1)")
            print_usage_hints()
            return
        time.sleep(1)
    sys.exit("Still not ready after 15 minutes. Check server.log; the computer may be out of memory.")


def cmd_stop():
    pid = load_state().get("pid")
    if not pid_alive(pid):
        save_state(pid=None)
        print("Nothing running.")
        return
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not pid_alive(pid):
                break
            time.sleep(0.5)
        else:
            os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    if pid_alive(pid):
        sys.exit(f"Couldn't stop pid {pid}. Close it manually.")
    save_state(pid=None, serving=None)
    if port_in_use():
        print(f"Stopped pid {pid}, but something else still uses port {ARGS.port}.")
    else:
        print(f"Stopped (pid {pid}); port {ARGS.port} is free.")


def cmd_status():
    state = load_state()
    keys = ("server", "build", "backend", "models", "active", "serving", "pid", "port")
    print(json.dumps({k: state.get(k) for k in keys}, indent=1))
    print(f"running ({state.get('serving')})" if healthy() else "not running")


# ask and chat

THINK_END = re.compile(r"</ifm\|think\w*>")


def read_message(m):
    """Return (thinking, answer, tool_calls) from a reply.

    llama.cpp normally splits these already. But 3.7B at medium effort opens with
    <ifm|think_fast> and closes with </ifm|think>, so the server never sees the end of
    the thinking and puts the whole reply in reasoning_content. Split it here instead.
    """
    reasoning = m.get("reasoning_content") or ""
    content = m.get("content") or ""
    if THINK_END.search(reasoning) or "<ifm|" in content:
        return split_output(reasoning + content)
    calls = []
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = fn.get("arguments")
        calls.append({"name": fn.get("name"), "arguments": args})
    return reasoning.strip(), content.strip(), calls


def split_output(raw):
    """Split raw K2 Horizon text into (thinking, answer, tool_calls)."""
    parts = THINK_END.split(raw, maxsplit=1)
    thinking, answer = (parts[0], parts[1]) if len(parts) == 2 else ("", raw)
    thinking = re.sub(r"<ifm\|think\w*>", "", thinking)
    calls = []
    for name, body in re.findall(r"<ifm\|tool_call>\s*([\w.\-]+)(.*?)</ifm\|tool_call>", answer, re.S):
        pairs = re.findall(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*<ifm\|arg_value>(.*?)</ifm\|arg_value>", body, re.S)
        calls.append({"name": name, "arguments": {k.strip(): v.strip() for k, v in pairs}})
    answer = re.sub(r"<ifm\|tool_calls>.*?(</ifm\|tool_calls>|$)", "", answer, flags=re.S)
    answer = re.sub(r"</?ifm\|think\w*>", "", answer).strip()
    return thinking.strip(), answer, calls


def chat_request(messages, effort, stream=False):
    serving = load_state().get("serving")
    temp = MODELS[key_size(serving)]["temp"] if serving else 1.0
    body = {
        "messages": messages,
        "temperature": temp,
        "top_p": 0.95,
        "stream": stream,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    # No limit by default: at high effort the thinking alone can run to thousands of tokens.
    if ARGS.max_tokens:
        body["max_tokens"] = ARGS.max_tokens
    req = urllib.request.Request(f"http://127.0.0.1:{ARGS.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)


def cmd_ask():
    if not healthy():
        sys.exit("The model isn't running. Run: serve")
    start = time.time()
    with chat_request([{"role": "user", "content": ARGS.prompt}], ARGS.effort) as r:
        res = json.loads(r.read())
    thinking, answer, calls = read_message(res["choices"][0]["message"])
    if ARGS.show_thinking and thinking:
        print("-- thinking --\n" + thinking + "\n")
    print("-- answer --\n" + (answer or "(empty)"))
    for call in calls:
        print(f"-- tool call -- {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})")

    serving = load_state().get("serving")
    tokens = res.get("usage", {}).get("completion_tokens", "?")
    line = f"\n{label(serving) if serving else 'model'}, {time.time() - start:.1f}s, {tokens} tokens"
    tps = res.get("timings", {}).get("predicted_per_second")
    if tps:
        line += f", {tps:.0f} tokens/s"
    print(line)


def stream_reply(messages, effort):
    """Print the answer as it streams in and return it. Shows "thinking..." until the answer starts."""
    out = sys.stdout
    reasoning, content, tps = "", "", None
    thinking_shown = answer_started = False
    clear = "\r" + " " * 20 + "\r"
    with chat_request(messages, effort, stream=True) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            tps = (chunk.get("timings") or {}).get("predicted_per_second") or tps
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]
                if not thinking_shown and not answer_started:
                    out.write("k2 > thinking...")
                    out.flush()
                    thinking_shown = True
            if delta.get("content"):
                if not answer_started:
                    out.write((clear if thinking_shown else "") + "k2 > ")
                    answer_started = True
                content += delta["content"]
                out.write(delta["content"])
                out.flush()
    _, answer, _ = read_message({"reasoning_content": reasoning, "content": content})
    if not answer_started:  # 3.7B at medium effort, see read_message
        out.write((clear if thinking_shown else "") + "k2 > " + (answer or "(empty)"))
    print(f"\n   ({tps:.0f} tokens/s)\n" if tps else "\n")
    return answer


def cmd_chat():
    if not healthy():
        sys.exit("The model isn't running. Run: serve")
    if ARGS.window:
        # Use this same Python, in case python3 isn't on the new window's PATH.
        python = f'"{sys.executable}"'
        opened = open_in_window(f"{command_hint(python)} chat --effort {ARGS.effort}", "chat")
        print("Opened the chat in a new terminal window." if opened else "Couldn't open a terminal window on this computer.")
        print(f"  To start it yourself, run in a terminal: {command_hint()} chat")
        return
    try:
        import readline  # noqa: F401  (arrow keys and history; not on Windows)
    except ImportError:
        pass

    serving = load_state().get("serving")
    effort = ARGS.effort
    print(f"K2 Horizon {label(serving) if serving else ''}, {effort} thinking, running on this computer")
    print("Type a message and press Enter. /new starts over, /effort low|medium|high changes the thinking, /exit quits.\n")
    history = []
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            break
        if text == "/new":
            history = []
            print("(started over)\n")
            continue
        if text.startswith("/effort"):
            choice = text.split()[1:]
            if choice and choice[0] in EFFORTS:
                effort = choice[0]
                print(f"(thinking: {effort})\n")
            else:
                print("Use /effort low, /effort medium or /effort high\n")
            continue
        history.append({"role": "user", "content": text})
        try:
            history.append({"role": "assistant", "content": stream_reply(history, effort)})
        except KeyboardInterrupt:
            history.pop()
            print("\n(stopped)\n")


def open_in_window(cmd, name, folder=None):
    """Run a command in a new terminal window. Returns True if a window opened."""
    system = platform.system()
    if system == "Darwin":
        # Terminal opens .command files without asking for any extra permission.
        path = os.path.join(home(), f"{name}.command")
        with open(path, "w") as f:
            f.write("#!/bin/sh\nclear\n" + (f'cd "{folder}"\n' if folder else "") + cmd + "\n")
        os.chmod(path, 0o755)
        return subprocess.run(["open", path]).returncode == 0
    if system == "Windows":
        path = os.path.join(home(), f"{name}.bat")
        with open(path, "w") as f:
            f.write("@echo off\r\ntitle K2 Horizon\r\n" + (f'cd /d "{folder}"\r\n' if folder else "") + cmd + "\r\n")
        os.startfile(path)
        return True
    for term in (["x-terminal-emulator", "-e"], ["gnome-terminal", "--"], ["konsole", "-e"], ["xterm", "-e"]):
        if shutil.which(term[0]):
            script = (f'cd "{folder}" && ' if folder else "") + cmd
            subprocess.Popen(term + ["sh", "-c", script], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    return False


# Ollama and OpenCode

def ollama_running():
    return subprocess.run(["ollama", "list"], capture_output=True).returncode == 0


def start_ollama():
    if not shutil.which("ollama"):
        sys.exit("Ollama isn't installed. Get it from https://ollama.com/download, then run this again.")
    out = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
    found = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout + out.stderr)
    if found and tuple(map(int, found.groups())) < OLLAMA_MIN:
        sys.exit(f"K2 Horizon needs Ollama {'.'.join(map(str, OLLAMA_MIN))} or newer, and this is {found.group(0)}. "
                 "Update it from https://ollama.com/download, then run this again.")
    if ollama_running():
        return
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **background_flags())
    for _ in range(30):
        time.sleep(1)
        if ollama_running():
            return
    sys.exit("Ollama is installed but didn't start. Open the Ollama app, then run this again.")


def ollama_model():
    """Pull IFM's model and return (local name, IFM tag).

    The local name is IFM's model with a 16k context, because Ollama can default to
    4,096 tokens, which is too short for high reasoning or a coding agent. IFM's
    listing already sets the recommended temperature and top_p.
    """
    start_ollama()
    size = ARGS.size
    if not size:
        sys.exit(f"Choose a size: {ARGS.cmd} --size 0.9B | 3.7B | 7B   (run check to see what fits)")
    quant = pick_quant(size)
    gb = MODELS[size]["gb"][quant]
    if fit(gb, total_ram_gb()) == "too big":
        print(f"Warning: {size} {quant} is too big for this computer's memory, so it will be very slow. "
              "Run check to see what fits.")

    # Q4_K_M -> ifm/k2-horizon:3.7b-q4_K_M, BF16 -> ifm/k2-horizon:3.7b-bf16
    tag = f"{OLLAMA_REPO}:{size.lower()}-" + ("bf16" if quant == "BF16" else "q" + quant[1:])
    name = f"k2-horizon-{size.lower()}" + ("-bf16" if quant == "BF16" else "")
    print(f"Getting {tag} through Ollama ({gb:.1f} GB; a rerun skips what's already downloaded) ...")
    if subprocess.run(["ollama", "pull", tag]).returncode != 0:
        sys.exit("Ollama couldn't download the model (see above). Check the internet connection and run this again.")

    modelfile = os.path.join(home(), f"{name}.Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM {tag}\nPARAMETER num_ctx {CONTEXT}\n")
    out = subprocess.run(["ollama", "create", name, "-f", modelfile], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("Ollama couldn't set up the model:\n" + (out.stdout + out.stderr)[-800:])
    save_state(ollama_model=name, ollama_tag=tag)
    print(f"Ready in Ollama as {name} (from {tag}, {CONTEXT:,}-token context).")
    return name, tag


def run_interactive(cmd, name, folder=None):
    """Open cmd in a new window with --window, otherwise run it here if this is a real terminal."""
    if ARGS.window:
        line = " ".join(f'"{a}"' if " " in a else a for a in cmd)
        if open_in_window(line, name, folder):
            print("Opened it in a new terminal window.")
        else:
            print("Couldn't open a terminal window on this computer.")
    elif sys.stdin.isatty():
        subprocess.run(cmd, cwd=folder)


def cmd_ollama():
    name, tag = ollama_model()
    print(f"  Chat: ollama run {name}   (inside: /set think high|medium|low, /bye to quit)")
    print(f'  API:  http://127.0.0.1:11434/v1  with model "{name}"')
    print(f"  Remove it: ollama rm {name} {tag}")
    run_interactive(["ollama", "run", name], "ollama-chat")


def cmd_opencode():
    if not shutil.which("opencode"):
        sys.exit("OpenCode isn't installed. Get it from https://opencode.ai/download "
                 "(or run: curl -fsSL https://opencode.ai/install | bash), then run this again in a new terminal.")
    if ARGS.size == "0.9B":
        print("Warning: 0.9B is too small for coding work. In testing it guessed a file path that didn't exist, "
              "while 3.7B and 7B did the same task fine.")
    folder = os.path.abspath(os.path.expanduser(ARGS.folder))
    if not os.path.isdir(folder):
        sys.exit(f"No such folder: {folder}")
    name, tag = ollama_model()
    print(f"  Start it again later, from your project folder: ollama launch opencode --model {name}")
    print(f"  Remove the model: ollama rm {name} {tag}")
    run_interactive(["ollama", "launch", "opencode", "--model", name, "-y"], "opencode", folder)


# Command line

def build_arg(value):
    if re.fullmatch(r"latest|tested|b\d+", value):
        return value
    raise argparse.ArgumentTypeError(f"use tested ({TESTED_BUILD}), latest, or a build number like {TESTED_BUILD}")


def main():
    global ARGS
    sys.stdout.reconfigure(line_buffering=True)  # show progress right away, even when piped

    # --dir and --port work before or after the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=argparse.SUPPRESS,
                        help="where everything is stored (default ~/k2-horizon-local, or $K2_DIR)")
    common.add_argument("--port", type=int, default=argparse.SUPPRESS, help=f"server port (default {PORT})")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def command(name):
        return sub.add_parser(name, parents=[common])

    def model_options(p, default_format="gguf"):
        p.add_argument("--size", choices=list(MODELS))
        p.add_argument("--format", default=default_format, choices=list(FORMATS))
        p.add_argument("--quant", choices=QUANTS, help="GGUF file to use (BF16 is the same as --format gguf-bf16)")

    command("check")

    p = command("install")
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"])
    p.add_argument("--build", type=build_arg,
                   help=f"tested ({TESTED_BUILD}, the default), latest, or a build number")
    p.add_argument("--force", action="store_true")

    model_options(command("download"))

    p = command("serve")
    p.add_argument("--size", choices=list(MODELS))
    p.add_argument("--format", choices=list(FORMATS), help="default: the last one used for that size")
    p.add_argument("--ctx", type=int, default=CONTEXT)

    p = command("ask")
    p.add_argument("prompt")
    p.add_argument("--effort", default="high", choices=EFFORTS, help="how much the model thinks first (default high)")
    p.add_argument("--max-tokens", type=int, help="limit the reply length (default: no limit)")
    p.add_argument("--show-thinking", action="store_true")

    p = command("chat")
    p.add_argument("--effort", default="high", choices=EFFORTS)
    p.add_argument("--max-tokens", type=int, help="limit each reply's length (default: no limit)")
    p.add_argument("--window", action="store_true", help="open the chat in a new terminal window")

    for name in ("ollama", "opencode"):
        p = command(name)
        model_options(p)
        p.add_argument("--window", action="store_true", help="open it in a new terminal window")
    p.add_argument("--folder", default=".", help="the project folder to work in (default: the current one)")

    command("status")
    command("stop")

    ARGS = parser.parse_args()
    ARGS.dir = getattr(ARGS, "dir", None)
    ARGS.port = getattr(ARGS, "port", PORT)
    commands = {
        "check": cmd_check, "install": cmd_install, "download": cmd_download, "serve": cmd_serve,
        "ask": cmd_ask, "chat": cmd_chat, "status": cmd_status, "stop": cmd_stop,
        "ollama": cmd_ollama, "opencode": cmd_opencode,
    }
    commands[ARGS.cmd]()


if __name__ == "__main__":
    main()
