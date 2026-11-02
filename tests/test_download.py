import hashlib
import importlib.util
import io
from pathlib import Path
import re
import sys
import pytest


@pytest.fixture
def downloader(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("download_frontend_test", scripts / "download_frontend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_resumes_ranges_keeps_data_and_skips_verified_file(downloader, tmp_path, monkeypatch):
    source = bytes(range(251)) * 41
    target = tmp_path / "xlsr2_300m.pt"
    expected_hash = hashlib.sha256(source).hexdigest()
    monkeypatch.setattr(downloader, "XLSR_BYTES", len(source))
    monkeypatch.setattr(downloader, "XLSR_SHA256", expected_hash)
    monkeypatch.setattr(sys, "argv", ["download", "--output", str(target), "--connections", "4"])
    # A previously downloaded prefix must be reused, never truncated or requested again.
    prefix = target.with_name(target.name + ".part0")
    prefix.write_bytes(source[:100])
    requests = []
    class Response(io.BytesIO):
        def __init__(self, data, headers, status=200):
            super().__init__(data)
            self.headers, self.status = headers, status
    def urlopen(request, timeout):
        requests.append(request)
        if request.get_method() == "HEAD":
            return Response(b"", {"Content-Length": str(len(source))})
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        return Response(source[start:end + 1], {"Content-Range": f"bytes {start}-{end}/{len(source)}"}, 206)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", urlopen)
    downloader.main()
    assert target.read_bytes() == source
    assert all(target.with_name(target.name + f".part{i}").exists() for i in range(4))
    assert any(request.get_header("Range") == f"bytes=100-{len(source)//4-1}" for request in requests)
    requests.clear()
    downloader.main()
    assert requests == []  # The second invocation performs no network request/download.


def test_download_checksum_failure_preserves_existing_data(downloader, tmp_path, monkeypatch):
    source = b"synthetic bytes; not a model"
    target = tmp_path / "xlsr2_300m.pt"
    target.write_bytes(b"existing partial download")
    monkeypatch.setattr(downloader, "XLSR_BYTES", len(source))
    monkeypatch.setattr(downloader, "XLSR_SHA256", "0" * 64)
    monkeypatch.setattr(sys, "argv", ["download", "--output", str(target), "--connections", "1"])
    class Response(io.BytesIO):
        def __init__(self, data, headers, status=200):
            super().__init__(data)
            self.headers, self.status = headers, status
    def urlopen(request, timeout):
        if request.get_method() == "HEAD":
            return Response(b"", {"Content-Length": str(len(source))})
        return Response(source, {"Content-Range": f"bytes 0-{len(source)-1}/{len(source)}"}, 206)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.main()
    assert target.read_bytes() == b"existing partial download"
    assert target.with_name(target.name + ".part0").read_bytes() == source
    assert target.with_name(target.name + ".download").read_bytes() == source
