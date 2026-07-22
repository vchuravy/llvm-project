add_flang_tool(fir-opt fir-opt.cpp)
llvm_update_compile_flags(fir-opt)

# Export fir-opt's symbols so MLIR dialect/pass plugins loaded via
# --load-dialect-plugin / --load-pass-plugin (e.g. Enzyme's FIREnzyme) can
# resolve MLIR/LLVM symbols against the host, exactly as mlir-opt does. This is
# a build-only change: it does not alter fir-opt's behavior.
export_executable_symbols_for_plugins(fir-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

if(FLANG_INCLUDE_TESTS)
  set(test_libs
    FIRTestAnalysis
    FIRTestOpenACCInterfaces
    MLIROpenACCTestPasses
    MLIRTestIR
    )
endif()

target_link_libraries(fir-opt PRIVATE
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRSupport
  FIRTransforms
  FIRCodeGen
  FIRCodeGenDialect
  HLFIRDialect
  HLFIRTransforms
  FIROpenACCSupport
  FIROpenACCTransforms
  FIROpenMPSupport
  FlangOpenMPTransforms
  FIRAnalysis
  MIFDialect
  flangPasses
  ${test_libs}
)

mlir_target_link_libraries(fir-opt PRIVATE
  ${dialect_libs}
  ${extension_libs}

  # TODO: these should be transitive dependencies from a target providing
  # "registerFIRPasses()"
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRFuncToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRVectorToLLVM
  MLIROptLib
)
