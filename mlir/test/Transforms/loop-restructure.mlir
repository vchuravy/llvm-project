// RUN: mlir-opt %s -loop-restructure -split-input-file | FileCheck %s

func @simple() {
  %ci1 = constant 1 : index
  %ci10 = constant 10 : index
  br ^bb0(%ci10)

  ^bb0(%i: index):
    %i1 = addi %i, %ci1: index
    %cmp = cmpi "eq", %i1, %ci10 : index
    cond_br %cmp, ^exit, ^bb1

  ^bb1:
    br ^bb0(%i1)

  // CHECK loop.natural (%i)
  // CHECK loop.natural.next {{.*}}: index
  // CHECK loop.natural.return

  ^exit:
    return
  
}