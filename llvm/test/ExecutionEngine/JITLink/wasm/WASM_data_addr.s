# Tests R_WASM_MEMORY_ADDR_SLEB: a function whose body contains an i32.const
# referencing a data symbol. After linking the 5-byte SLEB128 placeholder must
# be patched with the linear-memory address of my_data.
#
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: llvm-jitlink -triple=wasm32-unknown-emscripten -entry=get_addr -noexec -check %s %t.o

# get_addr's body starts with the i32.const opcode (0x41).
# jitlink-check: *{1}get_addr = 0x41

# my_data contains the 32-bit value 42 (0x2a) in little-endian.
# jitlink-check: *{1}my_data = 0x2a

.section .data,"",@
.globl my_data
my_data:
  .int32 42
  .size my_data, 4

.section .text,"",@
.globl get_addr
get_addr:
  .functype get_addr () -> (i32)
  i32.const my_data
  end_function
