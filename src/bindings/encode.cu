// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// encode.cu — perseus._core bindings of the device encoder (include/gpu_encode_bridge.h): tiles
// registered once per session, a giant group's diagonals encoded on the GPU at the level a product
// with a given ciphertext needs, products with those plaintexts through the same fhe->mult the CPU
// path uses (so the recorded op stream is unchanged), and eviction.
#include "fideslib_wrapper.h"
#include "gpu_encode_bridge.h"
#include "inference.h"
#include "packing/packed_ctx.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using PtxImpl = typename Ptx::element_type;

std::vector<double> rows_of(const py::array_t<double, py::array::c_style | py::array::forcecast>& W, int& d) {
    if (W.ndim() != 2 || W.shape(0) != W.shape(1))
        throw std::invalid_argument("register_tile: W must be a square 2-D array");
    d = static_cast<int>(W.shape(0));
    const double* p = W.data();
    return std::vector<double>(p, p + static_cast<size_t>(d) * d);
}

}  // namespace

void bind_encode(py::module_& m) {
    // The device plaintext as Python sees it: an opaque handle (its facade Plaintext). evict_ptx
    // frees the device copy and forgets the graph name and tag, and the handle is refused by
    // mult_ptx afterwards (filling.py evicts every group after its products). Dropping the last
    // reference instead does the same through the facade's destructor and the wrap's deleter.
    py::class_<PtxImpl, Ptx>(m, "DevicePlaintext")
        .def_property_readonly("level",
                               [](const PtxImpl& p) {
                                   if (!p.loaded || !p.cpu.has_value())
                                       throw std::invalid_argument("DevicePlaintext.level: the plaintext was evicted");
                                   return p.GetLevel();
                               })
        .def_property_readonly("loaded", [](const PtxImpl& p) { return p.loaded; });

    m.def(
        "gpu_encode_available", [](Inference& inf) { return inf.fhe && inf.cc() && inf.cc()->gpu.has_value(); },
        py::arg("inf"), "True when the session runs on a device the encoder can use.");

    m.def(
        "gpu_encode_fits",
        [](Inference& inf, const PackedCtx& ref) {
            py::gil_scoped_release nogil;
            return gpu_encode::bridge_for(inf)->fits(ref.ct);
        },
        py::arg("inf"), py::arg("ref"),
        "True when encode_group_gpu can serve a product with `ref` (a degree-1 encode); otherwise the "
        "caller takes the CPU route for that group.");

    m.def(
        "register_tile",
        [](Inference& inf, py::array_t<double, py::array::c_style | py::array::forcecast> W) {
            int d = 0;
            auto rows = rows_of(W, d);
            py::gil_scoped_release nogil;
            return gpu_encode::bridge_for(inf)->register_tile(rows, d);
        },
        py::arg("inf"), py::arg("W"),
        "Upload a d x d folded tile (row-major, y = x @ W) once; returns its tile id (invalid after "
        "release_tiles).");

    m.def(
        "tile_values",
        [](Inference& inf, uint64_t tile_id) {
            int d = 0;
            std::vector<double> W;
            {
                py::gil_scoped_release nogil;
                W = gpu_encode::bridge_for(inf)->tile_values(tile_id, d);
            }
            py::array_t<double> out({static_cast<py::ssize_t>(d), static_cast<py::ssize_t>(d)});
            std::memcpy(out.mutable_data(), W.data(), sizeof(double) * W.size());
            return out;
        },
        py::arg("inf"), py::arg("tile_id"),
        "The resident tile back on the host (row-major d x d): the CPU route's source when a group "
        "cannot take the device route.");

    m.def(
        "encode_group_gpu",
        [](Inference& inf, const std::vector<uint64_t>& tile_ids, const std::vector<int>& baby, int g, int s,
           const PackedCtx& ref, const std::vector<std::string>& names) {
            py::gil_scoped_release nogil;
            return gpu_encode::bridge_for(inf)->encode_group(tile_ids, baby, g, s, ref.ct, names);
        },
        py::arg("inf"), py::arg("tile_ids"), py::arg("baby"), py::arg("g"), py::arg("s"), py::arg("ref"),
        py::arg("names"),
        "Encode diagonals (g, baby[b]) of tiles[b] on the device at the level a product with `ref` "
        "needs; returns one DevicePlaintext per entry, named `names[b]` in the graph.");

    m.def(
        "mult_ptx",
        [](Inference& inf, const PackedCtx& ct, Ptx& pt) {
            if (!inf.fhe) throw std::runtime_error("mult_ptx: the session has no FHE runtime");
            if (!pt || !pt->loaded)
                throw std::invalid_argument("mult_ptx: the device plaintext was evicted or never made; encode the "
                                            "group again");
            py::gil_scoped_release nogil;
            return inf.fhe->mult(ct, pt);
        },
        py::arg("inf"), py::arg("ct"), py::arg("pt"),
        "ct * a device plaintext, through the same product the CPU path records.");

    m.def(
        "evict_ptx",
        [](Inference& inf, Ptx& pt) {
            py::gil_scoped_release nogil;
            gpu_encode::bridge_for(inf)->evict(pt);
        },
        py::arg("inf"), py::arg("pt"), "Free the device plaintext; the handle is refused by mult_ptx afterwards.");

    m.def(
        "gpu_encode_stats",
        [](Inference& inf) {
            const auto st = gpu_encode::bridge_for(inf)->stats();
            py::dict d;
            d["plaintexts"] = st.plaintexts;
            d["batches"] = st.batches;
            d["k1_ms"] = st.k1_ms;
            d["fft_ms"] = st.fft_ms;
            d["k2_ms"] = st.k2_ms;
            d["ntt_ms"] = st.ntt_ms;
            return d;
        },
        py::arg("inf"), "Counters of the device encoder (device-side stage times when FHE_GPU_ENCODE_TIMING=1).");

    m.def(
        "release_tiles",
        [](Inference& inf) {
            py::gil_scoped_release nogil;
            gpu_encode::bridge_for(inf)->release_tiles();
        },
        py::arg("inf"), "Free the resident tiles; their ids are refused from then on.");

    m.def(
        "drop_gpu_encoder",
        [](Inference& inf) {
            py::gil_scoped_release nogil;
            gpu_encode::drop_bridge(inf);
        },
        py::arg("inf"),
        "Release the session's device encoder (tiles, metadata plaintexts, streams, plans); Session.close "
        "calls it before the Inference goes away.");
}
