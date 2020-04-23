add_mlir_conversion_library(MLIRLinalgToLLVM
  LinalgToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  MLIRLoopLikeInterfaceIncGen
)

target_link_libraries(MLIRLinalgToLLVM
  PUBLIC
  MLIRAffineToStandard
  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRStandardToLLVM
  MLIRVectorToLLVM
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
