"""Perseus's numpy fake with the runtime's level rules on a composite chain.

``perseus.impl.fake`` charges a product's rescale at once: the output lands at degree 1,
``unit`` primes deeper, and refreshes when that level reaches the ceiling. The runtime leaves
the rescale pending instead (include/fideslib_wrapper.h):

* a product lands at degree 2 at its inputs' effective level, a degree-2 input counting
  ``unit`` deeper ("mult of deg-2@L and of deg-1@L+d both land deg-2@L+d", :1029-1034);
* the reactive refresh after a product or a subtraction tests the unrealized level
  (``maybe_bootstrap``, :2446);
* a bootstrap lands unrealized, at degree 2 (only the planned path realizes it, :992-999);
* adding an encoded plaintext realizes the pending rescale, a ciphertext sum keeps the deeper
  operand, and a fresh encryption is degree 1 at the landing level.

These are the rules the planner simulates (perseus/plan/placer/sim.py ``simulate``), so eager
counts taken here are the ones a GPU session would show. On the stock fake an operand one
product short of the ceiling cannot take that product; here it can, which moves every
``bootstrap_hint`` threshold by one level.
"""

from __future__ import annotations

from perseus.impl import fake


class RuntimeFakeFhe(fake.FakeFhe):
    def eff(self, a):
        return a.level + (self.unit if a.noise_deg == 2 else 0)

    def _land(self, vec, level, packing):
        out = fake.FakeCt(vec, level, 2, packing)
        if level >= self._limit and not self.auto_bts_suppressed:
            if self.strict:
                raise RuntimeError(f"reactive bootstrap at level {level} (strict fake)")
            self.n_reactive += 1
            self.bootstrap(out)
        return out

    def _sum(self, a, b, vec):
        if a.noise_deg == b.noise_deg:
            return fake.FakeCt(vec, max(a.level, b.level), a.noise_deg, a.packing)
        d2, d1 = (a, b) if a.noise_deg == 2 else (b, a)
        if self.eff(d2) > d1.level:
            return fake.FakeCt(vec, d2.level, 2, a.packing)
        return fake.FakeCt(vec, d1.level, 1, a.packing)

    def add(self, a, b):
        if isinstance(b, fake.FakeCt):
            return self._sum(a, b, a.vec + b.vec)
        return fake.FakeCt(a.vec + float(b), a.level, a.noise_deg, a.packing)

    def sub(self, a, b):
        if isinstance(b, fake.FakeCt):
            self.calls.append("sub_cc")
            out = self._sum(a, b, a.vec - b.vec)
            self.maybe_bootstrap(out)
            return out
        return fake.FakeCt(a.vec - float(b), a.level, a.noise_deg, a.packing)

    def inplace_add(self, a, b):
        out = self._sum(a, b, a.vec + b.vec)
        a.vec, a.level, a.noise_deg = out.vec, out.level, out.noise_deg

    def mult(self, a, b):
        if isinstance(b, fake.FakeCt):
            self.calls.append("mult_cc")
            return self._land(a.vec * b.vec, max(self.eff(a), self.eff(b)), a.packing)
        self.calls.append("mult_sc")
        return self._land(a.vec * float(b), self.eff(a), a.packing)

    def square(self, a):
        self.calls.append("square")
        return self._land(a.vec * a.vec, self.eff(a), a.packing)

    def bootstrap(self, a):
        super().bootstrap(a)
        a.noise_deg = 2


class RuntimeFakeInf(fake.FakeInf):
    def _times(self, a, vec):
        self.fhe.calls.append("mult_pt")
        return self.fhe._land(a.vec * vec, self.fhe.eff(a), a.packing)

    def _plus(self, a, vec):
        return fake.FakeCt(a.vec + vec, self.fhe.eff(a), 1, a.packing)

    def mult_pt(self, a, values):
        return self._times(a, self._full(values))

    def mult_slot_pt(self, a, name):
        return self._times(a, self.slot_pts[name])

    def mult_cached(self, a, tag, values, tagged=True):  # noqa: ARG002 (the runtime's signature)
        return self._times(a, self._cached(tag, a, values))

    def mult_const(self, a, re, im):
        self.fhe.calls.append("mult_const")
        return self.fhe._land(a.vec * complex(re, im), self.fhe.eff(a), a.packing)

    def add_pt(self, a, values):
        return self._plus(a, self._full(values))

    def add_slot_pt(self, a, name):
        return self._plus(a, self.slot_pts[name])

    def add_cached(self, a, tag, values, tagged=True):  # noqa: ARG002 (the runtime's signature)
        return self._plus(a, self._cached(tag, a, values))


def make_runtime_fake(
    N=1024,
    hid=32,
    dim=24,
    H=4,
    H_real=3,
    E=128,
    E_real=96,
    limit=46,
    bts_out=34,
    unit=2,
    strict=False,
    rot_steps=(),
    complex_payload=False,
    sparse_slots=(512, 1),
):
    """``fake.make_fake`` with the runtime's rules: (fhe, inf). Composite chains only (the
    plaintext-add rule of a one-prime level depends on the plaintext's level, not modelled)."""
    if unit < 2:
        raise ValueError("the runtime rules here are those of a composite chain (unit >= 2)")
    fhe = RuntimeFakeFhe(
        rot_steps,
        limit=limit,
        bts_out=bts_out,
        unit=unit,
        strict=strict,
        slots=N,
        complex_payload=complex_payload,
        sparse_slots=sparse_slots,
    )
    return fhe, RuntimeFakeInf(fhe, N, hid, dim, H, H_real, E, E_real)
