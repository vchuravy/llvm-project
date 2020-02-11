//===- LoopRestructure.cpp - Find natural Loops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "NaturalLoops"

namespace {
  struct LoopRestructure : public OperationPass<LoopRestructure> {
    LoopRestructure() = default;
    LoopRestructure(const LoopRestructure &) {}


    void runOnOperation() override;
  }

} // end anonymous namespace

void LoopRestructure::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  for (Region &region : getOperation()->getRegions()) {
  }
}


std::unique_ptr<Pass> mlir::createLoopRestructurePass() { return std::make_unique<LoopRestructure>(); }

static PassRegistration<LoopRestructure> pass("loop-restructure", "Lift natural loops in the CFG to NaturalLoopOps");
