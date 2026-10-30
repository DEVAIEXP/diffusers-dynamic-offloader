"""Release-draft boundary tests with in-memory wheels and mocked registries."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/prepare_github_release.py'
spec = importlib.util.spec_from_file_location('release_draft', SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
REPO = 'DEVAIEXP/diffusers-dynamic-offloader'
VERSION = '0.1.0rc4'
COMMIT = 'a' * 40


def wheel_bytes(version=VERSION):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as wheel:
        wheel.writestr('diffusers_dynamic_offloader.dist-info/METADATA',
                       f'Name: diffusers-dynamic-offloader\nVersion: {version}\n')
    return buffer.getvalue()


@pytest.mark.parametrize('repository,version,commit', [
    ('outside/repository', VERSION, COMMIT), (REPO, '0.1.0', COMMIT),
    (REPO, '0.1.0rc4;echo unsafe', COMMIT), (REPO, VERSION, 'main'),
])
def test_explicit_release_identity_is_required(repository, version, commit):
    with pytest.raises(ValueError):
        release.validate_request(repository, version, commit)


def test_wheel_hash_metadata_and_path_are_verified():
    data = wheel_bytes()
    digest = hashlib.sha256(data).hexdigest()
    release.verify_wheel(data, 'package.whl', 'diffusers-dynamic-offloader', VERSION, digest)
    for name, version, expected in [('package.whl', VERSION, '0' * 64),
                                    ('../package.whl', VERSION, digest),
                                    ('package.whl', '0.1.0rc5', digest)]:
        with pytest.raises(ValueError):
            release.verify_wheel(data, name, 'diffusers-dynamic-offloader', version, expected)


def test_draft_attaches_exact_bytes_and_never_publishes_or_creates_tag(tmp_path, monkeypatch):
    data = wheel_bytes()
    digest = hashlib.sha256(data).hexdigest()
    filename = 'diffusers_dynamic_offloader-0.1.0rc4-py3-none-any.whl'
    registry = {'releases': {VERSION: [{'filename': filename, 'digests': {'sha256': digest},
                  'url': 'https://files.pythonhosted.org/package.whl', 'packagetype': 'bdist_wheel', 'yanked': False}]}}
    monkeypatch.setattr(release, 'read_url', lambda url, limit, headers=None: json.dumps(registry).encode() if url.endswith('/json') else data)
    calls = []
    def api(path, token, method='GET', data=None, binary=False):
        calls.append((path, method, data))
        if '/commits/' in path:
            return {'sha': COMMIT}
        if method == 'GET':
            return []
        if path.endswith('/generate-notes'):
            return {'body': 'Generated changes'}
        if path.endswith('/releases'):
            assert data['draft'] is True and data['prerelease'] is True
            assert data['target_commitish'] == COMMIT
            return {'id': 1, 'html_url': 'https://github.com/example/draft',
                    'upload_url': 'https://uploads.github.com/example/assets{?name}'}
        return {'size': len(data), 'state': 'uploaded'}
    monkeypatch.setattr(release, 'api', api)
    output = tmp_path / 'new-release'
    release.prepare(REPO, VERSION, COMMIT, output, 'test-token')
    assert (output / filename).read_bytes() == data
    assert 'Generated changes' in (output / 'RELEASE_NOTES.md').read_text()
    assert all(method != 'PATCH' for _, method, _ in calls)
    assert not any('/git/refs' in path for path, _, _ in calls)
    assert len([path for path, _, _ in calls if path.startswith('https://uploads.github.com/')]) == 4


def test_existing_draft_is_not_replaced(tmp_path, monkeypatch):
    def api(path, *args, **kwargs):
        if '/commits/' in path:
            return {'sha': COMMIT}
        return [{'tag_name': 'v' + VERSION}] if path.endswith('per_page=100') else []
    monkeypatch.setattr(release, 'api', api)
    monkeypatch.setattr(release, 'download_wheels', lambda *args: pytest.fail('Must refuse before downloading'))
    with pytest.raises(ValueError, match='already exists'):
        release.prepare(REPO, VERSION, COMMIT, tmp_path / 'new-release', 'test-token')


def test_workflow_is_manual_and_cannot_run_gpu_or_publish_pypi():
    text = (SCRIPT.parents[1] / '.github/workflows/release-draft.yml').read_text()
    assert 'workflow_dispatch:' in text and 'workflow_call:' in text
    assert 'source_commit:' in text and 'persist-credentials: false' in text
    for forbidden in ['push:', 'pull_request:', 'twine', 'pip install', '--gpus', 'self-hosted']:
        assert forbidden not in text
