# SPDX-License-Identifier: MIT
"""Read a PyTorch ``.pth`` state dict without torch: the zip archive's ``data.pkl`` unpickled with the tensor
rebuild hooks replaced, each storage read from ``<archive>/data/<key>`` as numpy (bf16 widened to float32).

Only what ``torch.save(state_dict)`` writes is supported: ``_rebuild_tensor_v2`` over typed storages (the legacy
``FloatStorage`` / ``BFloat16Storage`` / ``HalfStorage`` names, or the ``torch.storage.UntypedStorage`` form with a
dtype)."""
from __future__ import annotations

import pickle
import zipfile

import numpy as np

_DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": "bf16",
       "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_,
       "float32": np.float32, "float16": np.float16, "bfloat16": "bf16", "float64": np.float64, "int64": np.int64}


class _Dtype:
    def __init__(self, name):
        self.name = name


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *_a, **_k):
    arr, = storage
    n = int(np.prod(size)) if size else 1
    if not size:
        return arr[offset].copy()
    itemstrides = tuple(int(s) for s in stride)
    view = np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * arr.itemsize for s in itemstrides))
    out = np.array(view)
    assert out.size == n
    return out


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch" and name in _DT:
            return _Dtype(name)
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *_a, **_k: data
        raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _tag, st, key, _loc, _numel = pid
        name = st.name if isinstance(st, (_StorageType, _Dtype)) else str(st)
        dt = _DT[name]
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            if dt == "bf16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=dt)
            self.cache[key] = arr
        return (self.cache[key],)


def load_state_dict(path: str) -> dict[str, np.ndarray]:
    """{name: float32 numpy array} of a ``torch.save``d state dict."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with zf.open(pkl) as f:
            sd = _Unpickler(f, zf, prefix).load()
    return {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}
