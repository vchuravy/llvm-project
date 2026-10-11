; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s --check-prefix=ON
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 --spirv-fp-contract=off %s -o - | FileCheck %s --check-prefix=OFF
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 --spirv-fp-contract=off %s -o - -filetype=obj | spirv-val %}

; opencl.enable.FP_CONTRACT allows contraction, so kernels get no default, unless
; --spirv-fp-contract=off forces it.

; ON-NOT: FloatControls2
; ON-NOT: FPFastMathDefault

; OFF-DAG: OpCapability FloatControls2
; OFF-DAG: OpExtension "SPV_KHR_float_controls2"
; OFF-DAG: OpExecutionModeId %[[#]] FPFastMathDefault %[[#FLOAT:]] %[[#ZERO:]]
; OFF-DAG: %[[#FLOAT]] = OpTypeFloat 32
; OFF-DAG: %[[#ZERO]] = OpConstantNull %[[#]]

define spir_kernel void @k(ptr addrspace(1) %p, float %x) {
  %y = fmul float %x, 2.0
  store float %y, ptr addrspace(1) %p
  ret void
}

!opencl.enable.FP_CONTRACT = !{}
