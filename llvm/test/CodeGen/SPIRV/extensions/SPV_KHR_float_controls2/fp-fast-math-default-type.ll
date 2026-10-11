; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s --check-prefix=COUNT
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; With FPFastMathDefault given through spirv.ExecutionMode, instructions
; without fast-math flags get the default for their floating-point type. That
; type must be found for virtual registers that MachineCSE creates, which have
; no type in the registry, and for OpFunctionCall, whose operand 2 is the callee.

; CHECK-DAG: OpName %[[#CALLEE:]] "callee"
; CHECK-DAG: OpDecorate %[[#CSE_A:]] FPFastMathMode None
; CHECK-DAG: OpDecorate %[[#CSE_D:]] FPFastMathMode None
; CHECK-DAG: OpDecorate %[[#CMP_A:]] FPFastMathMode None
; CHECK-DAG: OpDecorate %[[#CMP:]] FPFastMathMode None
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#CSE_A]] = OpFMul %[[#FLOAT]] %[[#]] %[[#]]
; CHECK-DAG: %[[#CSE_D]] = OpFMul %[[#FLOAT]] %[[#CSE_A]] %[[#]]
; CHECK-DAG: %[[#CMP_A]] = OpFMul %[[#FLOAT]] %[[#]] %[[#]]
; CHECK-DAG: %[[#CMP]] = OpFOrdLessThan %[[#]] %[[#CMP_A]] %[[#]]
; CHECK-DAG: OpFunctionCall %[[#]] %[[#CALLEE]] %[[#]]

; The integer call is not decorated.
; COUNT-COUNT-4: FPFastMathMode
; COUNT-NOT: FPFastMathMode

define spir_kernel void @cse(ptr addrspace(1) %p, float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  %a = fmul float %x, %y
  store float %a, ptr addrspace(1) %p
  br label %exit

exit:
  %b = fmul float %x, %y
  %d = fmul float %b, %y
  store float %d, ptr addrspace(1) %p
  ret void
}

define spir_kernel void @cse_cmp(ptr addrspace(1) %p, ptr addrspace(1) %q, float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  %a = fmul float %x, %y
  store float %a, ptr addrspace(1) %p
  br label %exit

exit:
  %b = fmul float %x, %y
  %cmp = fcmp olt float %b, %y
  %z = zext i1 %cmp to i32
  store i32 %z, ptr addrspace(1) %q
  ret void
}

define spir_func i32 @callee(i32 %x) noinline {
  %r = add i32 %x, 1
  ret i32 %r
}

define spir_kernel void @call(ptr addrspace(1) %p, i32 %a) {
  %r = call spir_func i32 @callee(i32 %a)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

!spirv.ExecutionMode = !{!0, !1, !2}
!0 = !{ptr @cse, i32 6028, float poison, i32 0}
!1 = !{ptr @cse_cmp, i32 6028, float poison, i32 0}
!2 = !{ptr @call, i32 6028, float poison, i32 0}
