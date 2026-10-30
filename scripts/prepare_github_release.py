"""Prepare a reviewable GitHub prerelease from the exact published PyPI wheels.

No builds, GPU execution, tag pushes, release publication or PyPI uploads occur.
The selected source commit must be the maintainer-verified build revision.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import zipfile
from email.parser import BytesParser

PROJECTS = {
    "DEVAIEXP/diffusers-dynamic-offloader": "diffusers-dynamic-offloader",
    "DEVAIEXP/ddo-vmm-allocator": "ddo-vmm-allocator",
}
MAX_WHEEL_BYTES = 200 * 1024 * 1024


def validate_request(repository, version, commit):
    if repository not in PROJECTS:
        raise ValueError("Unsupported release repository")
    if not re.fullmatch(r"\d+\.\d+\.\d+rc\d+", version):
        raise ValueError("An explicit rc prerelease version is required")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full reviewed source commit SHA is required")


def read_url(url, limit, headers=None):
    with urlopen(Request(url, headers=headers or {}), timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Response exceeds the fixed download limit")
    return data


def verify_wheel(data, filename, project, version, expected):
    if Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError("Unsafe or unexpected asset name")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Published wheel SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1 or wheel.getinfo(names[0]).file_size > 1024 * 1024:
            raise ValueError("Unexpected wheel metadata")
        metadata = BytesParser().parsebytes(wheel.read(names[0]))
    if metadata["Name"].replace("_", "-") != project or metadata["Version"] != version:
        raise ValueError("Wheel project/version mismatch")


def download_wheels(repository, version, output):
    project = PROJECTS[repository]
    # The project endpoint avoids a cached 404 from an earlier version lookup.
    registry = json.loads(read_url(f"https://pypi.org/pypi/{project}/json", 16 * 1024 * 1024))
    files = registry["releases"].get(version, [])
    if not files or any(item["packagetype"] != "bdist_wheel" or item["yanked"] for item in files):
        raise ValueError("Expected a non-yanked, wheel-only published version")
    if len(files) > 8:
        raise ValueError("Unexpected published asset count")
    records = []
    for item in files:
        url = urlsplit(item["url"])
        if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
            raise ValueError("Wheel must come from the official PyPI file host")
        data = read_url(item["url"], MAX_WHEEL_BYTES)
        verify_wheel(data, item["filename"], project, version, item["digests"]["sha256"])
        (output / item["filename"]).write_bytes(data)
        records.append({"file": item["filename"], "sha256": item["digests"]["sha256"], "url": item["url"]})
    return records


def api(path, token, method="GET", data=None, binary=False):
    url = path if path.startswith("https://uploads.github.com/") else "https://api.github.com/" + path
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "Content-Type": "application/octet-stream" if binary else "application/json"}
    payload = data if binary else json.dumps(data).encode() if data is not None else None
    with urlopen(Request(url, method=method, data=payload, headers=headers), timeout=60) as response:
        return json.load(response)


def prepare(repository, version, commit, output, token):
    validate_request(repository, version, commit)
    output.mkdir(parents=True, exist_ok=False)
    base = "repos/" + repository
    source = api(base + "/commits/" + commit, token)
    if source["sha"] != commit:
        raise ValueError("Source commit was not resolved exactly")
    tag = "v" + version
    # Fail before creating a duplicate draft, replacing assets or moving a tag.
    releases = api(base + "/releases?per_page=100", token)
    tags = api(base + "/git/matching-refs/tags/" + quote(tag, safe=""), token)
    if any(item["tag_name"] == tag for item in releases) or any(item["ref"] == "refs/tags/" + tag for item in tags):
        raise ValueError("Release or tag already exists; inspect it instead of retrying")
    files = download_wheels(repository, version, output)
    (output / "SHA256SUMS").write_text("".join(f"{item['sha256']}  {item['file']}\n" for item in files), encoding="utf-8")
    manifest = {"schema": 1, "version": version, "repository": repository,
                "source_commit": commit, "source_commit_selection": "maintainer-reviewed build revision",
                "files": files, "rebuilt": False, "cuda_executed": False, "pypi_upload": False}
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    notes = api(base + "/releases/generate-notes", token, "POST", {"tag_name": tag, "target_commitish": commit})
    body = ("Experimental prerelease prepared for maintainer review.\n\n"
            "Assets are downloaded from PyPI and verified against its SHA256 digests; no rebuild occurred. "
            "Review the source commit, changelog and selected platform/model validation before publishing.\n\n"
            f"Source: https://github.com/{repository}/commit/{commit}\n\n"
            "Documentation: https://devaiexp.github.io/diffusers-dynamic-offloader/\n\n"
            "Examples and benchmarks: https://devaiexp.github.io/ddo-examples/\n\n" + notes["body"])
    (output / "RELEASE_NOTES.md").write_text(body, encoding="utf-8")
    release = api(base + "/releases", token, "POST", {"tag_name": tag, "target_commitish": commit,
                  "name": tag, "body": body, "draft": True, "prerelease": True, "make_latest": "false"})
    # Persist the draft identity immediately if a later upload fails.
    (output / "draft-receipt.json").write_text(json.dumps({"id": release["id"], "url": release["html_url"]}, indent=2) + "\n")
    upload_url = release["upload_url"].split("{", 1)[0]
    for path in sorted(output.iterdir()):
        if path.name == "draft-receipt.json":
            continue
        asset = api(upload_url + "?name=" + quote(path.name, safe=""), token, "POST", path.read_bytes(), binary=True)
        if asset["size"] != path.stat().st_size or asset["state"] != "uploaded":
            raise ValueError("Incomplete asset upload; preserve the existing draft")
    print(json.dumps({"draft": True, "url": release["html_url"], "version": version, "assets": len(files) + 3}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, choices=sorted(PROJECTS))
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repository, args.version, args.source_commit, args.output, os.environ["GH_TOKEN"])


if __name__ == "__main__":
    main()
