//===-------------- WASM.cpp - JIT linker function for WASM -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// WASM jit-link function.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/WASM.h"

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromWasmObject(MemoryBufferRef ObjectBuffer,
                             std::shared_ptr<orc::SymbolStringPool> SSP) {
  return make_error<JITLinkError>("WASM support not implemented yet");
}

void link_Wasm(std::unique_ptr<LinkGraph> G,
               std::unique_ptr<JITLinkContext> Ctx) {
  Ctx->notifyFailed(make_error<JITLinkError>("WASM support not implemented yet"));
}

} // end namespace jitlink
} // end namespace llvm
