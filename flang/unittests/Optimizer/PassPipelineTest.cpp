//===- PassPipelineTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercise the HLFIR-stage extension points of the HLFIR-to-FIR pass pipeline.
// These callbacks let out-of-tree transformations (e.g. automatic
// differentiation) plug passes in while HLFIR intrinsic operations
// (hlfir.sum, hlfir.matmul, ...) are still present, before they are lowered
// to FIR/runtime calls. The callbacks run at pipeline-construction time, so no
// IR is needed: building the pipeline is enough to observe them.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Tools/CrossToolHelpers.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace {

// Both HLFIR extension-point callbacks are invoked, Early before Last, and the
// Early hook runs before any pass has been added to the pipeline.
TEST(HLFIRExtensionPoint, CallbacksAreInvokedInOrder) {
  mlir::MLIRContext context;
  mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
  MLIRToLLVMPassPipelineConfig config(llvm::OptimizationLevel::O2);

  std::vector<std::string> order;
  size_t earlySizeAtCall = ~size_t{0}; // sentinel

  config.registerHLFIROptEarlyEPCallbacks(
      [&](mlir::PassManager &nestedPm, llvm::OptimizationLevel) {
        order.push_back("early");
        earlySizeAtCall = nestedPm.size();
      });
  config.registerHLFIROptLastEPCallbacks(
      [&](mlir::PassManager &, llvm::OptimizationLevel) {
        order.push_back("last");
      });

  fir::createHLFIRToFIRPassPipeline(pm, fir::EnableOpenMP::None, config);

  ASSERT_EQ(order.size(), 2u);
  EXPECT_EQ(order[0], "early");
  EXPECT_EQ(order[1], "last");
  // Early is the very first thing in the pipeline: nothing added yet.
  EXPECT_EQ(earlySizeAtCall, 0u);
  // The pipeline added passes around the Last hook (before lowering intrinsics).
  EXPECT_GT(pm.size(), 0u);
}

// A callback may inject passes into the pipeline at the extension point.
TEST(HLFIRExtensionPoint, CallbackCanAddPasses) {
  mlir::MLIRContext context;
  mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
  MLIRToLLVMPassPipelineConfig config(llvm::OptimizationLevel::O0);

  size_t sizeBefore = ~size_t{0};
  size_t sizeAfter = ~size_t{0};
  config.registerHLFIROptEarlyEPCallbacks(
      [&](mlir::PassManager &nestedPm, llvm::OptimizationLevel) {
        sizeBefore = nestedPm.size();
        nestedPm.addPass(mlir::createCanonicalizerPass());
        sizeAfter = nestedPm.size();
      });

  fir::createHLFIRToFIRPassPipeline(pm, fir::EnableOpenMP::None, config);

  EXPECT_EQ(sizeBefore, 0u);
  EXPECT_EQ(sizeAfter, 1u);
}

// With no callbacks registered the pipeline is still built normally: the hooks
// are a no-op by default and do not perturb the default pipeline.
TEST(HLFIRExtensionPoint, NoCallbacksIsNoOp) {
  mlir::MLIRContext context;
  mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
  MLIRToLLVMPassPipelineConfig config(llvm::OptimizationLevel::O2);

  fir::createHLFIRToFIRPassPipeline(pm, fir::EnableOpenMP::None, config);

  EXPECT_GT(pm.size(), 0u);
}

} // namespace
