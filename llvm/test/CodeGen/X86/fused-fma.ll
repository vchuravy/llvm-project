; RUN: llc < %s -march=x86-64 -mcpu=haswell | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64--linux-gnu"

%jl_value_t = type opaque

;CHECK-LABEL: julia_dotf:
define double @julia_dotf(%jl_value_t addrspace(10)* nonnull align 16 dereferenceable(40), %jl_value_t addrspace(10)* nonnull align 16 dereferenceable(40)) {
top:
    %2 = addrspacecast %jl_value_t addrspace(10)* %0 to %jl_value_t addrspace(11)*
    %3 = bitcast %jl_value_t addrspace(11)* %2 to %jl_value_t addrspace(10)* addrspace(11)*
    %4 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(11)* %3, i64 3
    %5 = bitcast %jl_value_t addrspace(10)* addrspace(11)* %4 to i64 addrspace(11)*
    %6 = load i64, i64 addrspace(11)* %5, align 8
    %7 = icmp sgt i64 %6, 0
    br i1 %7, label %L13.lr.ph, label %L32

L13.lr.ph:                                        ; preds = %top
   %8 = bitcast %jl_value_t addrspace(11)* %2 to double addrspace(13)* addrspace(11)*
   %9 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %8, align 8
   %10 = addrspacecast %jl_value_t addrspace(10)* %1 to %jl_value_t addrspace(11)*
   %11 = bitcast %jl_value_t addrspace(11)* %10 to double addrspace(13)* addrspace(11)*
   %12 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %11, align 8
   %min.iters.check = icmp ult i64 %6, 16
   br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %L13.lr.ph
   %n.vec = and i64 %6, -16
   br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
    %vec.phi = phi <4 x double> [ zeroinitializer, %vector.ph ], [ %33, %vector.body ]
    %vec.phi10 = phi <4 x double> [ zeroinitializer, %vector.ph ], [ %34, %vector.body ]
    %vec.phi11 = phi <4 x double> [ zeroinitializer, %vector.ph ], [ %35, %vector.body ]
    %vec.phi12 = phi <4 x double> [ zeroinitializer, %vector.ph ], [ %36, %vector.body ]
    %13 = getelementptr inbounds double, double addrspace(13)* %9, i64 %index
    %14 = bitcast double addrspace(13)* %13 to <4 x double> addrspace(13)*
    %wide.load = load <4 x double>, <4 x double> addrspace(13)* %14, align 8
    %15 = getelementptr double, double addrspace(13)* %13, i64 4
    %16 = bitcast double addrspace(13)* %15 to <4 x double> addrspace(13)*
    %wide.load13 = load <4 x double>, <4 x double> addrspace(13)* %16, align 8
    %17 = getelementptr double, double addrspace(13)* %13, i64 8
    %18 = bitcast double addrspace(13)* %17 to <4 x double> addrspace(13)*
    %wide.load14 = load <4 x double>, <4 x double> addrspace(13)* %18, align 8
    %19 = getelementptr double, double addrspace(13)* %13, i64 12
    %20 = bitcast double addrspace(13)* %19 to <4 x double> addrspace(13)*
    %wide.load15 = load <4 x double>, <4 x double> addrspace(13)* %20, align 8
    %21 = getelementptr inbounds double, double addrspace(13)* %12, i64 %index
    %22 = bitcast double addrspace(13)* %21 to <4 x double> addrspace(13)*
    %wide.load16 = load <4 x double>, <4 x double> addrspace(13)* %22, align 8
    %23 = getelementptr double, double addrspace(13)* %21, i64 4
    %24 = bitcast double addrspace(13)* %23 to <4 x double> addrspace(13)*
    %wide.load17 = load <4 x double>, <4 x double> addrspace(13)* %24, align 8
    %25 = getelementptr double, double addrspace(13)* %21, i64 8
    %26 = bitcast double addrspace(13)* %25 to <4 x double> addrspace(13)*
    %wide.load18 = load <4 x double>, <4 x double> addrspace(13)* %26, align 8
    %27 = getelementptr double, double addrspace(13)* %21, i64 12
    %28 = bitcast double addrspace(13)* %27 to <4 x double> addrspace(13)*
    %wide.load19 = load <4 x double>, <4 x double> addrspace(13)* %28, align 8
;CHECK: vfmadd231pd
;CHECK: vfmadd231pd
;CHECK: vfmadd231pd
;CHECK: vfmadd231pd
    %29 = fmul contract <4 x double> %wide.load, %wide.load16
    %30 = fmul contract <4 x double> %wide.load13, %wide.load17
    %31 = fmul contract <4 x double> %wide.load14, %wide.load18
    %32 = fmul contract <4 x double> %wide.load15, %wide.load19
    %33 = fadd fast <4 x double> %vec.phi, %29
    %34 = fadd fast <4 x double> %vec.phi10, %30
    %35 = fadd fast <4 x double> %vec.phi11, %31
    %36 = fadd fast <4 x double> %vec.phi12, %32
    %index.next = add i64 %index, 16
    %37 = icmp eq i64 %index.next, %n.vec
    br i1 %37, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
    %bin.rdx = fadd fast <4 x double> %34, %33
    %bin.rdx20 = fadd fast <4 x double> %35, %bin.rdx
    %bin.rdx21 = fadd fast <4 x double> %36, %bin.rdx20
    %rdx.shuf = shufflevector <4 x double> %bin.rdx21, <4 x double> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
    %bin.rdx22 = fadd fast <4 x double> %bin.rdx21, %rdx.shuf
    %rdx.shuf23 = shufflevector <4 x double> %bin.rdx22, <4 x double> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
    %bin.rdx24 = fadd fast <4 x double> %bin.rdx22, %rdx.shuf23
    %38 = extractelement <4 x double> %bin.rdx24, i32 0
    %cmp.n = icmp eq i64 %6, %n.vec
   br i1 %cmp.n, label %L32, label %scalar.ph

scalar.ph:                                        ; preds = %middle.block, %L13.lr.ph
   %bc.resume.val = phi i64 [ %n.vec, %middle.block ], [ 0, %L13.lr.ph ]
   %bc.merge.rdx = phi double [ %38, %middle.block ], [ 0.000000e+00, %L13.lr.ph ]
   br label %L13

L13:                                              ; preds = %scalar.ph, %L13
   %value_phi16 = phi i64 [ %bc.resume.val, %scalar.ph ], [ %45, %L13 ]
   %value_phi5 = phi double [ %bc.merge.rdx, %scalar.ph ], [ %44, %L13 ]
    %39 = getelementptr inbounds double, double addrspace(13)* %9, i64 %value_phi16
    %40 = load double, double addrspace(13)* %39, align 8
    %41 = getelementptr inbounds double, double addrspace(13)* %12, i64 %value_phi16
    %42 = load double, double addrspace(13)* %41, align 8
    %43 = fmul contract double %40, %42
    %44 = fadd fast double %value_phi5, %43
    %45 = add nuw nsw i64 %value_phi16, 1
    %46 = icmp ult i64 %45, %6
   br i1 %46, label %L13, label %L32

L32:                                              ; preds = %L13, %middle.block, %top
   %value_phi2 = phi double [ 0.000000e+00, %top ], [ %44, %L13 ], [ %38, %middle.block ]
  ret double %value_phi2
}


