# Tests R_WASM_FUNCTION_INDEX_LEB: a direct call from one defined function to
# another. After linking the 5-byte LEB128 placeholder in the call instruction
# must be patched with the callee's function index.
#
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: llvm-jitlink -triple=wasm32-unknown-emscripten -entry=caller -noexec -check %s %t.o

# Caller's body starts with the call opcode (0x10).
# jitlink-check: *{1}caller = 0x10

# Callee's body is a no-op: just the end opcode (0x0b).
# jitlink-check: *{1}callee = 0xb

.functype callee () -> ()

.globl caller
caller:
  .functype caller () -> ()
  call callee
  end_function

.globl callee
callee:
  .functype callee () -> ()
  end_function
