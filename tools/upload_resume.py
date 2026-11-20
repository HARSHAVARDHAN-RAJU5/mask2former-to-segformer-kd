"""Copy one big file to the server over SSH, with a progress bar, that resumes after network cuts.

How it works:
  1. Ask the server how many bytes of the file it already has.
  2. Send the rest, starting from that point (appended to the server file).
  3. If the connection drops, wait and go back to step 1. Repeat until done.
  4. Compare SHA-256 checksums on both sides to prove the copy is exact.

While running, it asks Windows not to sleep (no settings are changed).

Usage:
  python tools/upload_resume.py <local_file> <user@host> <remote_path>
"""

import ctypes
import hashlib
import os
import subprocess
import sys
import time

from tqdm import tqdm

CHUNK = 4 * 1024 * 1024  # 4 MB per read
RETRY_WAIT = 30  # seconds to wait after a failure
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",  # notice a dead connection in about a minute
    "-o", "ServerAliveCountMax=4",
]


def keep_awake(on):
    """Ask Windows not to sleep while we run (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def ssh(host, command):
    return subprocess.run(["ssh", *SSH_OPTS, host, command], capture_output=True, text=True)


def remote_size(host, path):
    """Bytes already on the server (0 if the file isn't there). None if the server can't be reached."""
    r = ssh(host, f"stat -c %s '{path}' 2>/dev/null || echo 0")
    if r.returncode != 0:
        return None
    return int(r.stdout.strip() or 0)


def send_from(host, local, path, offset, bar):
    """Append local[offset:] to the server file. Returns True if everything was sent."""
    proc = subprocess.Popen(["ssh", *SSH_OPTS, host, f"cat >> '{path}'"], stdin=subprocess.PIPE)
    try:
        with open(local, "rb") as f:
            f.seek(offset)
            while chunk := f.read(CHUNK):
                proc.stdin.write(chunk)
                bar.update(len(chunk))
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    return proc.wait() == 0


def sha256_local(local):
    h = hashlib.sha256()
    with open(local, "rb") as f, tqdm(total=os.path.getsize(local), unit="B", unit_scale=True,
                                      desc="checksum (PC)") as bar:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    local, host, path = sys.argv[1:]
    if not path.startswith("/") or ":" in path:
        sys.exit(f"Remote path must be a Linux path like /home/... (got {path!r}). "
                 "In Git Bash, run with MSYS_NO_PATHCONV=1, or use PowerShell.")
    total = os.path.getsize(local)
    keep_awake(True)
    try:
        while True:
            done = remote_size(host, path)
            if done is None:
                print(f"\nServer not reachable. Retrying in {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                continue
            if done > total:
                sys.exit(f"Server file is bigger than the local file ({done} > {total}). Stopping; check it by hand.")
            if done == total:
                break
            with tqdm(total=total, initial=done, unit="B", unit_scale=True, unit_divisor=1024,
                      desc="upload", dynamic_ncols=True) as bar:
                ok = send_from(host, local, path, done, bar)
            if not ok:
                print(f"\nConnection dropped. Resuming in {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)

        print("Upload finished. Checking that both copies are identical...")
        local_hash = sha256_local(local)
        while (r := ssh(host, f"sha256sum '{path}'")).returncode != 0:
            print(f"Server not reachable for checksum. Retrying in {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
        remote_hash = r.stdout.split()[0]
        print(f"PC     : {local_hash}\nServer : {remote_hash}")
        print("OK: copy is exact." if local_hash == remote_hash else "MISMATCH: copy is damaged, do not use it.")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
