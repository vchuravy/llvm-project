; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; Without "spirv.ExecutionMode" or "opencl.enable.FP_CONTRACT" metadata, kernels
; get FPFastMathDefault without flags in place of the deprecated ContractionOff,
; for every floating-point type in the module. That requires the capability and
; the extension even when no instruction has fast-math flags.

; CHECK-DAG: OpCapability FloatControls2
; CHECK-DAG: OpExtension "SPV_KHR_float_controls2"
; CHECK-DAG: OpExecutionModeId %[[#STRICT:]] FPFastMathDefault %[[#FLOAT:]] %[[#ZERO:]]
; CHECK-DAG: OpExecutionModeId %[[#STRICT]] FPFastMathDefault %[[#DOUBLE:]] %[[#ZERO]]
; CHECK-DAG: OpExecutionModeId %[[#VECTOR:]] FPFastMathDefault %[[#FLOAT]] %[[#ZERO]]
; CHECK-NOT: ContractionOff
; CHECK-NOT: FPFastMathMode
; CHECK-DAG: %[[#FLOAT]] = OpTypeFloat 32
; CHECK-DAG: %[[#DOUBLE]] = OpTypeFloat 64
; CHECK-DAG: %[[#ZERO]] = OpConstantNull %[[#]]

define spir_kernel void @strict(ptr addrspace(1) %p, ptr addrspace(1) %q, float %x, double %y) {
  %a = fmul float %x, 2.0
  store float %a, ptr addrspace(1) %p
  %b = fadd double %y, 1.0
  store double %b, ptr addrspace(1) %q
  ret void
}

; a floating-point type only in the signature, and in a vector
define spir_kernel void @vector(ptr addrspace(1) %p, <4 x float> %x) {
  store i32 1, ptr addrspace(1) %p
  ret void
}
