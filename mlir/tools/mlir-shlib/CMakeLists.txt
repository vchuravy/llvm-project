# This tool creates a shared library from the MLIR libraries. Generating this
# library is enabled by setting MLIR_BUILD_MLIR_DYLIB=yes on the CMake
# commandline.

set(SOURCES
  libmlir.cpp
  )

#if(MLIR_BUILD_MLIR_DYLIB)

set(LIBS
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRLoopsToGPU
  MLIRLinalgToLLVM

  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRGPUtoNVVMTransforms
  MLIRGPUtoROCDLTransforms
  MLIRGPUtoSPIRVTransforms
  MLIRLinalgOps
  MLIRLLVMIR
  MLIRLoopOps
  MLIRNVVMIR
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  MLIRROCDLIR
  MLIRSPIRV
  MLIRStandardToSPIRVTransforms
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
)
if(MLIR_CUDA_CONVERSIONS_ENABLED)
  list(APPEND LIBS
    MLIRGPUtoCUDATransforms
  )
endif()

add_llvm_library(MLIR SHARED DISABLE_LLVM_LINK_LLVM_DYLIB SONAME ${INSTALL_WITH_TOOLCHAIN} ${SOURCES})

llvm_update_compile_flags(MLIR)
whole_archive_link(MLIR ${LIBS})
target_link_libraries(MLIR PRIVATE ${LIBS})
#endif()

