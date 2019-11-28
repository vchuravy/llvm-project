; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+reference-types | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i8 addrspace(256)* @test(i8 addrspace(256)*) 


; CHECK-LABEL: call_test:
; CHECK: .functype       call_test (anyref) -> (anyref)
define i8 addrspace(256)* @call_test(i8 addrspace(256)*) {
; CHECK: anyref.call     $push0=, test, $0
  %a = call i8 addrspace(256)* @test(i8 addrspace(256)* %0) 
  ret i8 addrspace(256)* %a
}

; TODO: nullref?
; define i8 addrspace(256)* @null_test() {
;   ret i8 addrspace(256)* null
; }

; TODO: Loading a anyref from a pointer
; @glob = external global i8 addrspace(256)*, align 4
; define i8 addrspace(256)* @global_test() {
;   %a = load i8 addrspace(256)*, i8 addrspace(256)** @glob
;   ret i8 addrspace(256)* %a
; }

; CHECK: .functype       test (anyref) -> (anyref)
