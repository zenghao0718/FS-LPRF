"""Download the official Fairseq file byte-for-byte (no Hugging Face conversion)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import _bootstrap
from fs_lprf.provenance import XLSR_URL, XLSR_BYTES, XLSR_SHA256

URL = XLSR_URL


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="pretrained/xlsr2_300m.pt")
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.connections <= 16:
        parser.error("--connections must be between 1 and 16")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == XLSR_BYTES:
        if sha256_file(target) != XLSR_SHA256:
            raise ValueError("Existing file has an unexpected SHA-256; it has been preserved")
        print(json.dumps(dict(status="already_verified", url=URL, bytes=XLSR_BYTES, sha256=XLSR_SHA256)), flush=True)
        return
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    if size != XLSR_BYTES:
        raise ValueError("Official artifact size changed; existing data has been preserved")
    def fetch(part):
        start = size * part // args.connections
        end = size * (part + 1) // args.connections - 1
        path = target.with_name(target.name + f".part{part}")
        for attempt in range(6):
            offset = path.stat().st_size if path.exists() else 0
            if offset == end - start + 1:
                return path
            if offset > end - start + 1:
                raise ValueError(f"Invalid partial download: {path}")
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start + offset}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("Official server did not honor Range request")
                    if response.headers.get("Content-Range") != f"bytes {start + offset}-{end}/{size}":
                        raise RuntimeError("Official server returned an unexpected byte range")
                    with path.open("ab") as handle:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                if path.stat().st_size != end - start + 1:
                    raise OSError(f"Incomplete download: {path}")
                break
            except (OSError, urllib.error.URLError):
                if attempt == 5:
                    raise
                print(f"Retrying part {part + 1} (attempt {attempt + 2})", flush=True)
                time.sleep(min(2 ** attempt, 10))
        print(f"Part {part + 1}/{args.connections} complete", flush=True)
        return path
    print(f"Official checkpoint: {size} bytes, {args.connections} connections", flush=True)
    # Limit concurrent TLS handshakes on Windows; completed segments remain resumable.
    with ThreadPoolExecutor(max_workers=min(args.connections, 4)) as executor:
        parts = list(executor.map(fetch, range(args.connections)))
    temporary = target.with_name(target.name + ".download")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
                    output.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError("Downloaded size mismatch")
    if digest.hexdigest() != XLSR_SHA256:
        raise ValueError("Downloaded SHA-256 mismatch; all downloaded data has been preserved")
    temporary.replace(target)
    # Keep completed download segments so an interrupted/corrupt assembly can be resumed.
    provenance = dict(url=URL, bytes=size, sha256=digest.hexdigest())
    target.with_suffix(target.suffix + ".source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
