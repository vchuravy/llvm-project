# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: llvm-jitlink -triple=wasm32-unknown-emscripten -entry=entry -noexec -check %s %t.o

.globl  entry
entry:
  .functype entry () -> ()
  end_function
