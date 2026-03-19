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
#include "JITLinkGeneric.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::object;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

namespace {

static const char *getEdgeKindName(Edge::Kind K) {
  return getGenericEdgeKindName(K);
}

// Maps a WASM symbol's binding/visibility flags to JITLink Linkage/Scope.
static std::pair<Linkage, Scope> flagsToLinkageAndScope(uint32_t Flags) {
  Linkage L;
  switch (Flags & wasm::WASM_SYMBOL_BINDING_MASK) {
  case wasm::WASM_SYMBOL_BINDING_WEAK:
    L = Linkage::Weak;
    break;
  default:
    L = Linkage::Strong;
    break;
  }

  Scope S;
  if (Flags & wasm::WASM_SYMBOL_BINDING_LOCAL)
    S = Scope::Local;
  else if (Flags & wasm::WASM_SYMBOL_VISIBILITY_HIDDEN)
    S = Scope::Hidden;
  else
    S = Scope::Default;

  return {L, S};
}

class WasmLinkGraphBuilder {
public:
  WasmLinkGraphBuilder(const WasmObjectFile &Obj,
                       std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT)
      : Obj(Obj), SSP(std::move(SSP)), TT(std::move(TT)) {}

  Expected<std::unique_ptr<LinkGraph>> buildGraph() {
    G = std::make_unique<LinkGraph>(std::string(Obj.getFileName()),
                                    std::move(SSP), TT, SubtargetFeatures(),
                                    getEdgeKindName);

    if (auto Err = buildFunctions())
      return std::move(Err);
    if (auto Err = buildDataSegments())
      return std::move(Err);
    if (auto Err = buildSymbols())
      return std::move(Err);
    if (auto Err = addRelocations())
      return std::move(Err);

    return std::move(G);
  }

private:
  const WasmObjectFile &Obj;
  std::shared_ptr<orc::SymbolStringPool> SSP;
  Triple TT;
  std::unique_ptr<LinkGraph> G;

  // Function index (in combined import+defined space) → Block*
  DenseMap<uint32_t, Block *> FuncIndexToBlock;
  // Data segment index → Block*
  DenseMap<uint32_t, Block *> SegIndexToBlock;

  Section &getOrCreateSection(StringRef Name, orc::MemProt Prot) {
    if (auto *Sec = G->findSectionByName(Name))
      return *Sec;
    return G->createSection(Name, Prot);
  }

  Error buildFunctions() {
    if (Obj.functions().empty())
      return Error::success();

    Section &TextSec = getOrCreateSection(
        ".text", orc::MemProt::Read | orc::MemProt::Exec);

    uint32_t FuncIdx = Obj.getNumImportedFunctions();
    for (const auto &Func : Obj.functions()) {
      auto &B =
          G->createContentBlock(TextSec, ArrayRef<char>(
                                    reinterpret_cast<const char *>(Func.Body.data()),
                                    Func.Body.size()),
                                orc::ExecutorAddr(FuncIdx),
                                /*Alignment=*/1, /*AlignmentOffset=*/0);
      FuncIndexToBlock[FuncIdx] = &B;
      LLVM_DEBUG(dbgs() << "  Function[" << FuncIdx << "] → Block@"
                        << format_hex(FuncIdx, 10) << " size=" << Func.Body.size()
                        << "\n");
      ++FuncIdx;
    }
    return Error::success();
  }

  Error buildDataSegments() {
    uint32_t SegIdx = 0;
    for (const auto &Seg : Obj.dataSegments()) {
      // Passive segments have no memory address; skip for now.
      if (Seg.Data.InitFlags & wasm::WASM_DATA_SEGMENT_IS_PASSIVE) {
        ++SegIdx;
        continue;
      }
      Section &DataSec = getOrCreateSection(
          ".data", orc::MemProt::Read | orc::MemProt::Write);
      // Use segment index as a placeholder address; the memory manager will
      // assign real addresses during allocation.
      auto &B = G->createContentBlock(
          DataSec, ArrayRef<char>(
              reinterpret_cast<const char *>(Seg.Data.Content.data()),
              Seg.Data.Content.size()),
          orc::ExecutorAddr(SegIdx),
          /*Alignment=*/1u << Seg.Data.Alignment, /*AlignmentOffset=*/0);
      SegIndexToBlock[SegIdx] = &B;
      ++SegIdx;
    }
    return Error::success();
  }

  Error buildSymbols() {
    uint32_t SymIdx = 0;
    for (auto &SymRef : Obj.symbols()) {
      const auto &WS = Obj.getWasmSymbol(SymRef);
      auto [Lnk, Scp] = flagsToLinkageAndScope(WS.Info.Flags);

      if (WS.isTypeFunction()) {
        uint32_t FuncIdx = WS.Info.ElementIndex;
        if (WS.isDefined()) {
          auto It = FuncIndexToBlock.find(FuncIdx);
          if (It == FuncIndexToBlock.end())
            return make_error<JITLinkError>(
                "Function symbol " + WS.Info.Name +
                " references unknown function index " + Twine(FuncIdx));
          auto &Sym = G->addDefinedSymbol(*It->second, /*Offset=*/0,
                                          WS.Info.Name, It->second->getSize(),
                                          Lnk, Scp, /*IsCallable=*/true,
                                          /*IsLive=*/false);
          (void)Sym;
          LLVM_DEBUG(dbgs() << "  Defined function symbol: " << WS.Info.Name
                            << " → func[" << FuncIdx << "]\n");
        } else {
          G->addExternalSymbol(WS.Info.Name, /*Size=*/0,
                               /*IsWeaklyReferenced=*/Lnk == Linkage::Weak);
          LLVM_DEBUG(dbgs() << "  External function symbol: " << WS.Info.Name
                            << "\n");
        }
      } else if (WS.isTypeData()) {
        if (WS.isDefined()) {
          uint32_t SegIdx = WS.Info.DataRef.Segment;
          auto It = SegIndexToBlock.find(SegIdx);
          if (It == SegIndexToBlock.end())
            return make_error<JITLinkError>(
                "Data symbol " + WS.Info.Name +
                " references unknown segment " + Twine(SegIdx));
          G->addDefinedSymbol(*It->second, WS.Info.DataRef.Offset,
                               WS.Info.Name, WS.Info.DataRef.Size,
                               Lnk, Scp, /*IsCallable=*/false,
                               /*IsLive=*/false);
        } else {
          G->addExternalSymbol(WS.Info.Name, /*Size=*/0,
                               /*IsWeaklyReferenced=*/Lnk == Linkage::Weak);
        }
      }
      // Global, table, tag, section symbols: skip for now.
      ++SymIdx;
    }
    return Error::success();
  }

  Error addRelocations() {
    // Iterate over sections that carry relocations.
    for (auto &SecRef : Obj.sections()) {
      const auto &WasmSec = Obj.getWasmSection(SecRef);
      if (WasmSec.Relocations.empty())
        continue;
      // TODO: convert WasmRelocation entries to JITLink Edges.
      // For now, bail out if we encounter relocations so that any test
      // exercising them reports a clear error rather than silently ignoring
      // them.
      return make_error<JITLinkError>(
          "WASM relocations not yet implemented (section " +
          Twine(WasmSec.Type) + ")");
    }
    return Error::success();
  }
};

class WasmJITLinker : public JITLinker<WasmJITLinker> {
  friend class JITLinker<WasmJITLinker>;

public:
  WasmJITLinker(std::unique_ptr<JITLinkContext> Ctx,
                std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return make_error<JITLinkError>("WASM fixup not yet implemented");
  }
};

} // anonymous namespace

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromWasmObject(MemoryBufferRef ObjectBuffer,
                              std::shared_ptr<orc::SymbolStringPool> SSP) {
  Error Err = Error::success();
  auto ObjFile = std::make_unique<WasmObjectFile>(ObjectBuffer, Err);
  if (Err)
    return std::move(Err);

  LLVM_DEBUG(dbgs() << "Building LinkGraph from WASM object \""
                    << ObjectBuffer.getBufferIdentifier() << "\"\n");

  Triple TT;
  TT.setArch(ObjFile->getArch());
  TT.setObjectFormat(Triple::Wasm);

  return WasmLinkGraphBuilder(*ObjFile, std::move(SSP), TT).buildGraph();
}

void link_Wasm(std::unique_ptr<LinkGraph> G,
               std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;

  if (Ctx->shouldAddDefaultTargetPasses(G->getTargetTriple())) {
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple()))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  WasmJITLinker::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // end namespace jitlink
} // end namespace llvm
