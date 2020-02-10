// RUN: mlir-opt %s -loop-invariant-code-motion -split-input-file | FileCheck %s

// This should LICM out domove and domove only
//  * dontmove should not be moved as it is not guaranteed to execute. It is safe to move, but may result in unnecessary work (for example if an expensive function rather than addi were called)
//  * definitelydontmove sohuld not be moved as it is not loop invariant

// Note that this test requires loop.if to contain definitelydontmove to ensure the entire loop.if isn't LICM'd out (since then loop.if is loop invariant)

func @variant_loop_dialect(%cond : i1) {
  %ci0 = constant 0 : index
  %ci10 = constant 10 : index
  %ci1 = constant 1 : index
  loop.for %arg0 = %ci0 to %ci10 step %ci1 {
    loop.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = addi %arg0, %arg1 : index
      loop.if %cond {
         %dontmove = addi %ci0, %ci10 : index
      } else {
         %definitelydontmove = addi %arg0, %v0: index
      }
      %domove2 = muli %ci0, %ci1 : index
    }
  }

  // CHECK: muli
  // CHECK-NEXT: loop.for
  // CHECK-NEXT: loop.for
  // CHECK: addi
  // CHECK: addi
  // CHECK: addi

  return
}


func @variant_natloop_dialect(%cond : i1) {
  %ci0 = constant 0 : index
  %ci10 = constant 10 : index
  %ci1 = constant 1 : index

  loop.natural (%i) (%ci0 : index) {
         %in = addi %i, %ci1: index
         cond_br %cond, ^truebb, ^falsebb

      ^truebb:
         %dontmove = addi %ci0, %ci10 : index
         br ^falsebb

      ^falsebb:
         %domove2 = muli %ci0, %ci1 : index
         %cmp = cmpi "eq", %i, %ci10 : index
         cond_br %cmp, ^bb2, ^bb1

      ^bb1:
        loop.natural.next %in : index

      ^bb2:
        loop.natural.return
  }

  // CHECK: muli
  // CHECK-NEXT: loop.natural
  // CHECK: addi
  // CHECK: addi

  return
}
