//===------- WASM.h - Generic JIT link function for WASM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic jit-link functions for WASM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_WASM_H
#define LLVM_EXECUTIONENGINE_JITLINK_WASM_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

/// Create a LinkGraph from a WASM relocatable object.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromWasmObject(MemoryBufferRef ObjectBuffer,
                             std::shared_ptr<orc::SymbolStringPool> SSP);

/// jit-link the given WASM LinkGraph.
void link_Wasm(std::unique_ptr<LinkGraph> G,
               std::unique_ptr<JITLinkContext> Ctx);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_WASM_H
