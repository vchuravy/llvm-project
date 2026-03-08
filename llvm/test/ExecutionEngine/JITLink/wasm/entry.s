# RUN: llvm-mc -filetype=obj -position-independent -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-jitlink -entry=entry -noexec -check %s %t.o

.globl  entry
entry:
  .functype entry () -> ()
  end_function
