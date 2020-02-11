//===- LoopRestructure.cpp - Find natural Loops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopLikeInterface.h"
#include "mlir/Transforms/SideEffectsInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "NaturalLoops"

namespace {
  struct LoopRestructure : public OperationPass<LoopRestructure> {
    LoopRestructure() = default;
    LoopRestructure(const LoopRestructure &) {}


    void runOnOperation() override;
  };

} // end anonymous namespace
//===----------------------------------------------------------------------===//
 /// Stable LoopInfo Analysis - Build a loop tree using stable iterators so the
 /// result does / not depend on use list (block predecessor) order.
 ///
 

void regionMagic(DominanceInfo &domInfo, Region& region);

void LoopRestructure::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();

  for (Region &region : getOperation()->getRegions()) {
	regionMagic(domInfo, region);
  }

}

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopInfoImpl.h"

namespace mlir {
	class Loop : public llvm::LoopBase<mlir::Block, mlir::Loop> {
	private:
		Loop() = default;
		friend class llvm::LoopBase<Block, Loop>;
		friend class llvm::LoopInfoBase<Block, Loop>;
		explicit Loop(Block* B) : llvm::LoopBase<Block, Loop>(B) {}
		~Loop() = default;
	};
	class LoopInfo : public llvm::LoopInfoBase<mlir::Block, mlir::Loop> {
	public:
		LoopInfo(const llvm::DominatorTreeBase<mlir::Block, false> &DomTree) { analyze(DomTree); }
	};
}

template class llvm::LoopBase<::mlir::Block, ::mlir::Loop>;
template class llvm::LoopInfoBase<::mlir::Block, ::mlir::Loop>;




void regionMagic(DominanceInfo &domInfo, Region& region) {
   assert(domInfo.dominanceInfos.count(&region) != 0);
   auto DT = domInfo.dominanceInfos[&region].get();

   // Postorder traversal of the dominator tree.
   auto DomRoot = domInfo.getRootNode(&region);

   mlir::LoopInfo LI(*DT);

   for(auto L : LI) {
     llvm::errs() << " found mlir loop " << *L << "\n";
   }

}


std::unique_ptr<Pass> mlir::createLoopRestructurePass() { return std::make_unique<LoopRestructure>(); }

static PassRegistration<LoopRestructure> pass("loop-restructure", "Lift natural loops in the CFG to NaturalLoopOps");
