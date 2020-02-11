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

#include "mlir/Dialect/LoopOps/LoopOps.h"

using namespace mlir;

#define DEBUG_TYPE "NaturalLoops"

namespace {
  struct LoopRestructure : public OperationPass<LoopRestructure> {
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
  getOperation()->walk([&](Operation *op) {
	llvm::errs() << "running on operation: " << op << "\n";

  for (Region &region : op->getRegions()) {
	regionMagic(domInfo, region);
  }
  });


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
   llvm::errs() << "calling for region: " << &region << "\n";
   for(auto L : LI) {
     llvm::errs() << " found mlir loop " << *L << "\n";
    
	 // Create a caller block that will contain the loop op

	 // Set branches into loop (header) to branch into caller block

     // Create loop operation in caller block

     // Move loop header and loop blocks into loop operation

     // Replace branch to exit block with a new block that calls loop.natural.terminate
     //  In caller block, branch to correct exit block

     // For each back edge create a new block and replace the destination of that edge with said new block
     //    in that new block call llvm.natural.nextiteration 

     // Rewrite IV's

     // Rewrite values used outside of loop

     // Assert we only have one exit Later: Create new block for each exit

     // Create block for 
	 //OpBuilder builder(L->getPreheader());
	 L->getHeader();
   }

}

std::unique_ptr<Pass> mlir::createLoopRestructurePass() { return std::make_unique<LoopRestructure>(); }

static PassRegistration<LoopRestructure> pass("loop-restructure", "Lift natural loops in the CFG to NaturalLoopOps");
