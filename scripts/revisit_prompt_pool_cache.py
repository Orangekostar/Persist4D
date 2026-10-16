"""Exact, private CPU memoization of already validated official state pools.

Every lookup hashes the complete ordered input, taxonomy bytes and pinned code
and runtime identity. A miss calls the supplied original pool. This helper does
not resample references, combine process differences or change metric math.
Use one serial writer, followed by read-only consumers.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _metrics_valid(value):
    return (isinstance(value, dict) and len(value) == 6
            and all(isinstance(key, str) and (item is None or type(item) is float and math.isfinite(item))
                    for key, item in value.items()))


class OfficialPoolCache:
    def __init__(self, directory, *, pool, source_paths, runtime_identity, read_only=False):
        self.directory = Path(directory)
        self.pool = pool
        self.read_only = read_only
        self.hits = self.misses = 0
        self.sources = {name: Path(path).resolve() for name, path in source_paths.items()}
        self.sources.update(cache_implementation=Path(__file__).resolve(),
                            pool_implementation=Path(inspect.getsourcefile(pool)).resolve())
        self.provenance = dict(sources={name: dict(path=str(path), sha256=_file_sha(path))
                                       for name, path in sorted(self.sources.items())},
                               runtime=json.loads(_canonical(runtime_identity)))

    def _check_sources(self):
        for name, path in self.sources.items():
            if _file_sha(path) != self.provenance['sources'][name]['sha256']:
                raise RuntimeError('official pool cache source identity changed: ' + name)

    def __call__(self, states, *, dataset_spec):
        self._check_sources()
        states = list(states)
        taxonomy_path = Path(dataset_spec).resolve()
        taxonomy = dict(path=str(taxonomy_path), sha256=_file_sha(taxonomy_path))
        # Serialize all fields, including raw tensor bytes and the claimed state
        # checksum. Never trust the claimed checksum alone or sort reference order.
        key = _digest(_canonical(dict(schema=1, provenance=self.provenance,
                                      taxonomy=taxonomy, states=states)))
        path = self.directory / (key + '.json')
        if path.exists():
            try:
                payload = json.loads(path.read_bytes())
                checksum = payload.pop('checksum')
                valid = (checksum == _digest(_canonical(payload))
                         and payload['schema'] == 1 and payload['key'] == key
                         and payload['provenance'] == self.provenance
                         and payload['taxonomy'] == taxonomy and _metrics_valid(payload['metrics']))
                if not valid:
                    raise ValueError('payload differs')
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError('official pool cache integrity differs: ' + path.name) from error
            self.hits += 1
            return dict(payload['metrics'])

        self.misses += 1
        metrics = self.pool(states, dataset_spec=dataset_spec)
        if not _metrics_valid(metrics):
            raise ValueError('official pool returned invalid six-metric values')
        self._check_sources()
        if _file_sha(taxonomy_path) != taxonomy['sha256']:
            raise RuntimeError('official pool taxonomy changed during computation')
        if not self.read_only:
            payload = dict(schema=1, key=key, provenance=self.provenance,
                           taxonomy=taxonomy, metrics=metrics)
            payload['checksum'] = _digest(_canonical(payload))
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, prefix='.pool-', delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(_canonical(payload) + b'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return dict(metrics)
