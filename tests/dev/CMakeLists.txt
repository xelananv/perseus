# Dev/experimental test suite. Auto-discovers dev/test_*.cu and reuses the
# add_cachemir_test() helper from the parent tests/CMakeLists.txt. The parent
# tests/ dir (test_helpers.h) is already on the include path; add this dev/ dir
# too, since the family-scoped helper headers now live here.
file(GLOB CACHEMIR_DEV_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu")
foreach(src ${CACHEMIR_DEV_TESTS})
    get_filename_component(name ${src} NAME_WE)
    add_cachemir_test(${name})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# test_rotation_ext_moddown reaches the FIDESlib-native GPU API (Ciphertext::rotate(idx, moddown),
# GenCryptoContextGPU, ...) which is NOT part of the installed OpenFHE-compat headers, so it needs
# the FIDESlib source tree on its include path. Scoped to this one target.
# Device-encoder groundwork: a plain executable (no gtest) probing FIDESlib's limb order and the
# device-plaintext wrap on the library itself (examples/fhe_ssm_from_primitives's GPU encoder).
add_executable(gpu_encode_layout_probe gpu_encode_layout_probe.cu)
target_link_libraries(gpu_encode_layout_probe PRIVATE cachemir_lib)
target_include_directories(gpu_encode_layout_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
set_target_properties(gpu_encode_layout_probe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(gpu_encode_layout_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp -lineinfo -g>)   # host line info for the probe's backtraces

# The encoder's parity probe (COEFF integers, EVAL words, a product) — same shape as the layout probe.
add_executable(gpu_encode_probe gpu_encode_probe.cu)
target_link_libraries(gpu_encode_probe PRIVATE cachemir_lib)
target_include_directories(gpu_encode_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
set_target_properties(gpu_encode_probe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(gpu_encode_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp -lineinfo -g>)

if(TARGET test_rotation_ext_moddown)
    target_include_directories(test_rotation_ext_moddown PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
endif()

# test_ks_digit_intt calls FIDESlib::CKKS::setKsDigitIntt (the runtime A/B override for the
# redundant per-digit source INTT), declared in the FIDESlib-native Ciphertext.cuh.
if(TARGET test_ks_digit_intt)
    target_include_directories(test_ks_digit_intt PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
endif()

# The device magnitude probe of the graph capture against the host path's numbers (gpu_mag_probe.h).
add_executable(gpu_mag_probe_check gpu_mag_probe_check.cu)
target_link_libraries(gpu_mag_probe_check PRIVATE cachemir_lib)
target_include_directories(gpu_mag_probe_check PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party/FIDESlib/src)
set_target_properties(gpu_mag_probe_check PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(gpu_mag_probe_check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
