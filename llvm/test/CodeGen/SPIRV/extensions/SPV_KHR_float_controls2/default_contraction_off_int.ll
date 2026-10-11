; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; Without floating-point types, there is no FPFastMathDefault to emit, and
; nothing that requires the extension.

; CHECK-NOT: FloatControls2
; CHECK-NOT: SPV_KHR_float_controls2
; CHECK-NOT: FPFastMathDefault

define spir_kernel void @integer(ptr addrspace(1) %p, i32 %x) {
  %y = mul i32 %x, 2
  store i32 %y, ptr addrspace(1) %p
  ret void
}
