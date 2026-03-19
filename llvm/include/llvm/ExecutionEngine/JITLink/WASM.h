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

/// Edge kinds for WASM relocations.
namespace wasm {

enum EdgeKind_wasm : Edge::Kind {
  // R_WASM_FUNCTION_INDEX_LEB: direct call target; 5-byte ULEB128 function
  // index patched at the call site.
  FunctionIndexLEB = Edge::FirstRelocation,
  // R_WASM_TABLE_INDEX_SLEB: element/table reference; 5-byte SLEB128.
  TableIndexSLEB,
  // R_WASM_TABLE_INDEX_I32: element/table reference; 4-byte i32.
  TableIndexI32,
  // R_WASM_MEMORY_ADDR_LEB: linear memory address; 5-byte ULEB128.
  MemoryAddrLEB,
  // R_WASM_MEMORY_ADDR_SLEB: linear memory address; 5-byte SLEB128.
  MemoryAddrSLEB,
  // R_WASM_MEMORY_ADDR_I32: linear memory address; 4-byte i32.
  MemoryAddrI32,
  // R_WASM_TYPE_INDEX_LEB: call_indirect type signature; 5-byte ULEB128.
  TypeIndexLEB,
  // R_WASM_GLOBAL_INDEX_LEB: global index; 5-byte ULEB128.
  GlobalIndexLEB,
  // R_WASM_FUNCTION_INDEX_I32: function index as 4-byte i32.
  FunctionIndexI32,
  // R_WASM_TABLE_NUMBER_LEB: table number; 5-byte ULEB128.
  TableNumberLEB,
};

const char *getEdgeKindName(Edge::Kind K);

} // namespace wasm

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
