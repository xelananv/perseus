# SPDX-License-Identifier: MIT
"""The client-separated key path of run_planned (``--keys`` / ``--keyspec`` / ``--keygen``) on stand-ins for the
device build's ``perseus._pc`` (the CPU build has no Session, Client or bundle functions): which key source a flag
selects, the client role's files, that a bundle run encrypts and decrypts through the client object and never through
the session, the key request's model part, and the z-shard chain check that runs before a session opens."""
import json
import types

import numpy as np
import pytest

from perseus import _pc
from perseus.examples.rwkv7 import run_planned as rp


class _Secret:
    def __init__(self, b=b"s3cret"):
        self.b = bytes(b)

    @classmethod
    def generate(cls):
        return cls()

    @classmethod
    def from_bytes(cls, xs):
        return cls(bytes(xs))

    def to_bytes(self):
        return list(self.b)


class _Client:
    made = []

    def __init__(self, secret, complex=True):
        self.secret, self.calls = secret, []
        _Client.made.append(self)

    def keygen(self, spec, path, threads=8):
        open(path, "wb").write(b"bundle:" + spec.encode())
        return {"keys": len(json.loads(spec)["keys"])}

    def encrypt(self, v, level, deg):
        self.calls.append(("encrypt", level, deg))
        return b"frame:" + np.asarray(v).tobytes()

    def decrypt(self, frame):
        self.calls.append(("decrypt",))
        return [np.frombuffer(frame[len(b"ct:"):], dtype=np.complex128)]


@pytest.fixture
def fake_pc(monkeypatch):
    _Client.made = []
    monkeypatch.setattr(_pc, "Client", _Client, raising=False)
    monkeypatch.setattr(_pc, "ClientSecret", _Secret, raising=False)
    monkeypatch.setattr(_pc, "bundle_info", lambda p: {"bytes": 7, "contract": "c", "entries": {}}, raising=False)
    monkeypatch.setattr(_pc, "import_ct", lambda sess, frame: ("imported", frame), raising=False)
    monkeypatch.setattr(_pc, "export_ct", lambda sess, ct: b"ct:" + np.asarray(ct, dtype=np.complex128).tobytes(),
                        raising=False)
    return _pc


def test_resolve_keys(tmp_path, monkeypatch, fake_pc):
    assert rp.resolve_keys(types.SimpleNamespace(keys="inproc")) == (None, None)
    with pytest.raises(SystemExit, match="no keys.pckb"):
        rp.resolve_keys(types.SimpleNamespace(keys=str(tmp_path)))
    (tmp_path / "keys.pckb").write_bytes(b"x")
    with pytest.raises(SystemExit, match="client_secret.bin"):
        rp.resolve_keys(types.SimpleNamespace(keys=str(tmp_path)))
    (tmp_path / "client_secret.bin").write_bytes(b"abc")
    monkeypatch.setenv("RWKV7_KEYS", str(tmp_path))
    bundle, client = rp.resolve_keys(types.SimpleNamespace(keys="bundle"))
    assert bundle == tmp_path / "keys.pckb" and isinstance(client, _Client) and client.secret.b == b"abc"


def test_keygen_writes_bundle_and_secret(tmp_path, fake_pc):
    spec = tmp_path / "keyspec.json"
    spec.write_text(json.dumps({"model": {}, "session": {"keys": [1, 2, 3]}}))
    out = tmp_path / "keys"
    assert rp.main(["--keygen", str(out), "--keyspec-file", str(spec)]) == 0
    assert (out / "keys.pckb").read_bytes().startswith(b"bundle:")
    assert (out / "client_secret.bin").read_bytes() == b"s3cret"


def test_keyspec_needs_a_session_request(tmp_path, fake_pc):
    spec = tmp_path / "keyspec.json"
    spec.write_text(json.dumps({"model": {}, "session": None}))
    with pytest.raises(SystemExit, match="no session key request"):
        rp.main(["--keygen", str(tmp_path / "k"), "--keyspec-file", str(spec)])


def test_bundle_run_uses_the_client(fake_pc):
    """encrypt / decrypt go through the client object on a bundle run; inproc keys use the session's own."""
    run = rp.Rwkv7PlannedRun.__new__(rp.Rwkv7PlannedRun)
    cl = _Client(_Secret())
    run.client, run.bundle, run.sess = cl, "bundle.pckb", object()
    v = np.arange(4, dtype=np.float64)
    ct = run.encrypt(v, 30, 2)
    assert ct[0] == "imported" and cl.calls == [("encrypt", 30, 2)]
    got = run.decrypt(np.arange(3, dtype=np.complex128))
    assert np.array_equal(got, np.arange(3)) and cl.calls[-1] == ("decrypt",)
    assert run.key_mode() == "bundle"
    run2 = rp.Rwkv7PlannedRun.__new__(rp.Rwkv7PlannedRun)
    run2.client, run2.bundle = None, "bundle.pckb"
    with pytest.raises(RuntimeError, match="no secret"):
        run2.decrypt(None)
    run3 = rp.Rwkv7PlannedRun.__new__(rp.Rwkv7PlannedRun)
    run3.client, run3.bundle = None, None
    run3.fhe = types.SimpleNamespace(encrypt=lambda v, lvl, deg: ("session", lvl, deg))
    assert run3.encrypt(v, 3, 1) == ("session", 3, 1) and run3.key_mode() == "insecure-inproc"


def test_key_needs_and_keyspec_on_cpu(tmp_path):
    """--keyspec's model part on the CPU build (the session's request needs the device build: noted, not faked)."""
    from perseus.examples.rwkv7 import ref
    from perseus.examples.rwkv7.session import make_model
    from perseus.examples.rwkv7.synth import make_state_dict
    P = ref.Params.from_state_dict(make_state_dict(2048, 1, vocab=64))
    sites = {}
    for n in ("ln1", "ln2"):
        sites[f"blocks.0.{n}.rsqrt"] = {"scale": 1.0, "lo": 0.1, "hi": 1.0, "eps": 1e-5}
    sites["blocks.0.att.kk_rsqrt"] = {"scale": 1.0, "lo": 0.1, "hi": 1.0, "eps": 0.0}
    sites["blocks.0.att.gn_rsqrt"] = {"scale": 1.0, "lo": 0.1, "hi": 1.0, "eps": 6.4e-4}
    _f, _i, m = make_model(P, sites)
    need = rp.key_needs(m)
    assert need["width"] == 2048 and need["shards"] == 1 and need["pcmm_t"] == [16]
    assert need["n_rotation_steps"] == len(need["rotation_steps"]) > 10
    out = tmp_path / "ks.json"
    assert rp.write_keyspec(str(out), None, None, {}, m) == 0
    doc = json.loads(out.read_text())
    assert doc["model"]["pcmm_t"] == [16] and doc["session"] is None and "device build" in doc["note"]


def test_shard_chain_check():
    """S > 1: a block whose output shard lands at another (level, deg) than the next block compiled for it stops the
    run before the session opens."""
    run = rp.Rwkv7PlannedRun.__new__(rp.Rwkv7PlannedRun)
    prog = lambda outs, ins: types.SimpleNamespace(outputs=outs, inputs=ins)   # noqa: E731
    run.labels = ["block_0", "block_1"]
    run.ref = {0: None, 1: None}
    run.reg = {0: {"out": [(0, 10), (1, 11)], "kv": []}}
    run.zin = {0: [0, 3], 1: [0, 5]}
    run.progs = {"block_0": prog([("o0", 26, 2), ("o1", 26, 2)], []),
                 "block_1": prog([], [("i0", 26, 2)] + [("x", 0, 0)] * 4 + [("i5", 26, 2)])}
    run.check_shard_chain()
    run.progs["block_1"].inputs[5] = ("i5", 24, 2)
    with pytest.raises(RuntimeError, match="do not chain"):
        run.check_shard_chain()
