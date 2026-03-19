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
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::object;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

namespace wasm {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case FunctionIndexLEB:
    return "FunctionIndexLEB";
  case TableIndexSLEB:
    return "TableIndexSLEB";
  case TableIndexI32:
    return "TableIndexI32";
  case MemoryAddrLEB:
    return "MemoryAddrLEB";
  case MemoryAddrSLEB:
    return "MemoryAddrSLEB";
  case MemoryAddrI32:
    return "MemoryAddrI32";
  case TypeIndexLEB:
    return "TypeIndexLEB";
  case GlobalIndexLEB:
    return "GlobalIndexLEB";
  case FunctionIndexI32:
    return "FunctionIndexI32";
  case TableNumberLEB:
    return "TableNumberLEB";
  default:
    return getGenericEdgeKindName(K);
  }
}

} // namespace wasm

namespace {

static const char *edgeKindName(Edge::Kind K) {
  return wasm::getEdgeKindName(K);
}

// Maps a WASM symbol's binding/visibility flags to JITLink Linkage/Scope.
static std::pair<Linkage, Scope> flagsToLinkageAndScope(uint32_t Flags) {
  Linkage L;
  switch (Flags & llvm::wasm::WASM_SYMBOL_BINDING_MASK) {
  case llvm::wasm::WASM_SYMBOL_BINDING_WEAK:
    L = Linkage::Weak;
    break;
  default:
    L = Linkage::Strong;
    break;
  }

  Scope S;
  if (Flags & llvm::wasm::WASM_SYMBOL_BINDING_LOCAL)
    S = Scope::Local;
  else if (Flags & llvm::wasm::WASM_SYMBOL_VISIBILITY_HIDDEN)
    S = Scope::Hidden;
  else
    S = Scope::Default;

  return {L, S};
}

// Map a WASM relocation type to a JITLink edge kind.
static Expected<Edge::Kind> wasmRelocToEdgeKind(uint8_t RelocType) {
  switch (RelocType) {
  case llvm::wasm::R_WASM_FUNCTION_INDEX_LEB:
    return wasm::FunctionIndexLEB;
  case llvm::wasm::R_WASM_TABLE_INDEX_SLEB:
    return wasm::TableIndexSLEB;
  case llvm::wasm::R_WASM_TABLE_INDEX_I32:
    return wasm::TableIndexI32;
  case llvm::wasm::R_WASM_MEMORY_ADDR_LEB:
    return wasm::MemoryAddrLEB;
  case llvm::wasm::R_WASM_MEMORY_ADDR_SLEB:
    return wasm::MemoryAddrSLEB;
  case llvm::wasm::R_WASM_MEMORY_ADDR_I32:
    return wasm::MemoryAddrI32;
  case llvm::wasm::R_WASM_TYPE_INDEX_LEB:
    return wasm::TypeIndexLEB;
  case llvm::wasm::R_WASM_GLOBAL_INDEX_LEB:
    return wasm::GlobalIndexLEB;
  case llvm::wasm::R_WASM_FUNCTION_INDEX_I32:
    return wasm::FunctionIndexI32;
  case llvm::wasm::R_WASM_TABLE_NUMBER_LEB:
    return wasm::TableNumberLEB;
  default:
    return make_error<JITLinkError>("Unsupported WASM relocation type " +
                                   Twine(RelocType));
  }
}

class WasmLinkGraphBuilder {
public:
  WasmLinkGraphBuilder(const WasmObjectFile &Obj,
                       std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT)
      : Obj(Obj), SSP(std::move(SSP)), TT(std::move(TT)) {}

  Expected<std::unique_ptr<LinkGraph>> buildGraph() {
    G = std::make_unique<LinkGraph>(std::string(Obj.getFileName()),
                                    std::move(SSP), TT, SubtargetFeatures(),
                                    edgeKindName);

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

  struct FuncInfo {
    Block *B;
    uint64_t BodyOffset; // byte offset of Body[0] within the code section content
  };

  // Function index (import+defined space) → FuncInfo
  DenseMap<uint32_t, FuncInfo> FuncIndexToInfo;
  // Data segment index → Block*
  DenseMap<uint32_t, Block *> SegIndexToBlock;
  // WASM symbol table index → JITLink Symbol*
  DenseMap<uint32_t, Symbol *> SymIndexToSym;
  // JITLink Symbol* → WASM function index (for FunctionIndexLEB fixups)
  DenseMap<Symbol *, uint32_t> SymToFuncIdx;

  Section &getOrCreateSection(StringRef Name, orc::MemProt Prot) {
    if (auto *Sec = G->findSectionByName(Name))
      return *Sec;
    return G->createSection(Name, Prot);
  }

  Error buildFunctions() {
    if (Obj.functions().empty())
      return Error::success();

    // Find the code section content so we can compute body offsets.
    ArrayRef<uint8_t> CodeSecContent;
    for (const auto &SecRef : Obj.sections()) {
      const auto &WasmSec = Obj.getWasmSection(SecRef);
      if (WasmSec.Type == llvm::wasm::WASM_SEC_CODE) {
        CodeSecContent = WasmSec.Content;
        break;
      }
    }

    Section &TextSec = getOrCreateSection(
        ".text", orc::MemProt::Read | orc::MemProt::Exec);

    uint32_t FuncIdx = Obj.getNumImportedFunctions();
    for (const auto &Func : Obj.functions()) {
      auto &B = G->createContentBlock(
          TextSec,
          ArrayRef<char>(reinterpret_cast<const char *>(Func.Body.data()),
                         Func.Body.size()),
          orc::ExecutorAddr(FuncIdx),
          /*Alignment=*/1, /*AlignmentOffset=*/0);
      // BodyOffset is the offset of Body[0] within the code section content.
      // Relocation offsets are also relative to the code section content start.
      uint64_t BodyOffset =
          CodeSecContent.empty()
              ? 0
              : (uint64_t)(Func.Body.data() - CodeSecContent.data());
      FuncIndexToInfo[FuncIdx] = {&B, BodyOffset};
      LLVM_DEBUG(dbgs() << "  Function[" << FuncIdx << "] bodyOffset=0x"
                        << Twine::utohexstr(BodyOffset)
                        << " size=" << Func.Body.size() << "\n");
      ++FuncIdx;
    }
    return Error::success();
  }

  Error buildDataSegments() {
    uint32_t SegIdx = 0;
    for (const auto &Seg : Obj.dataSegments()) {
      if (Seg.Data.InitFlags & llvm::wasm::WASM_DATA_SEGMENT_IS_PASSIVE) {
        ++SegIdx;
        continue;
      }
      Section &DataSec = getOrCreateSection(
          ".data", orc::MemProt::Read | orc::MemProt::Write);
      auto &B = G->createContentBlock(
          DataSec,
          ArrayRef<char>(
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

      Symbol *Sym = nullptr;
      if (WS.isTypeFunction()) {
        uint32_t FuncIdx = WS.Info.ElementIndex;
        if (WS.isDefined()) {
          auto It = FuncIndexToInfo.find(FuncIdx);
          if (It == FuncIndexToInfo.end())
            return make_error<JITLinkError>(
                "Function symbol " + WS.Info.Name +
                " references unknown function index " + Twine(FuncIdx));
          Sym = &G->addDefinedSymbol(*It->second.B, /*Offset=*/0, WS.Info.Name,
                                     It->second.B->getSize(), Lnk, Scp,
                                     /*IsCallable=*/true, /*IsLive=*/false);
          SymToFuncIdx[Sym] = FuncIdx;
          LLVM_DEBUG(dbgs() << "  Defined function symbol: " << WS.Info.Name
                            << " → func[" << FuncIdx << "]\n");
        } else {
          Sym = &G->addExternalSymbol(
              WS.Info.Name, /*Size=*/0,
              /*IsWeaklyReferenced=*/Lnk == Linkage::Weak);
          SymToFuncIdx[Sym] = FuncIdx;
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
          Sym = &G->addDefinedSymbol(*It->second, WS.Info.DataRef.Offset,
                                     WS.Info.Name, WS.Info.DataRef.Size, Lnk,
                                     Scp, /*IsCallable=*/false,
                                     /*IsLive=*/false);
        } else {
          Sym = &G->addExternalSymbol(
              WS.Info.Name, /*Size=*/0,
              /*IsWeaklyReferenced=*/Lnk == Linkage::Weak);
        }
      }
      // Global, table, tag, section symbols: skip for now.

      if (Sym)
        SymIndexToSym[SymIdx] = Sym;
      ++SymIdx;
    }
    return Error::success();
  }

  // Find the FuncInfo for the function whose body contains the given code
  // section offset.
  FuncInfo *findFunctionForOffset(uint64_t SectionOffset) {
    for (auto &[Idx, Info] : FuncIndexToInfo) {
      uint64_t BodyEnd = Info.BodyOffset + Info.B->getSize();
      if (Info.BodyOffset <= SectionOffset && SectionOffset < BodyEnd)
        return &Info;
    }
    return nullptr;
  }

  Error addRelocations() {
    for (auto &SecRef : Obj.sections()) {
      const auto &WasmSec = Obj.getWasmSection(SecRef);
      if (WasmSec.Relocations.empty())
        continue;

      for (const auto &Reloc : WasmSec.Relocations) {
        if (WasmSec.Type == llvm::wasm::WASM_SEC_CODE) {
          if (auto Err = addCodeRelocation(Reloc))
            return Err;
        } else {
          return make_error<JITLinkError>(
              "WASM relocations in non-code section " +
              Twine(WasmSec.Type) + " not yet implemented");
        }
      }
    }
    return Error::success();
  }

  Error addCodeRelocation(const llvm::wasm::WasmRelocation &Reloc) {
    auto EKOrErr = wasmRelocToEdgeKind(Reloc.Type);
    if (!EKOrErr)
      return EKOrErr.takeError();
    Edge::Kind EK = *EKOrErr;

    // Find which function body block contains this relocation.
    FuncInfo *Info = findFunctionForOffset(Reloc.Offset);
    if (!Info)
      return make_error<JITLinkError>(
          "WASM code relocation at offset " + Twine(Reloc.Offset) +
          " does not fall within any function body");

    uint64_t BlockOffset = Reloc.Offset - Info->BodyOffset;

    // For type-index relocations the Index is a type index, not a symbol
    // index. Represent the target as an absolute symbol with the type index
    // as its address so applyFixup can use the generic value path.
    if (EK == wasm::TypeIndexLEB || EK == wasm::TableNumberLEB) {
      auto &AbsSym = G->addAbsoluteSymbol(
          G->intern("__wasm_type_or_table_" + Twine(Reloc.Index).str()),
          orc::ExecutorAddr(Reloc.Index), /*Size=*/0, Linkage::Strong,
          Scope::Local, /*IsLive=*/true);
      Info->B->addEdge(EK, BlockOffset, AbsSym, Reloc.Addend);
      return Error::success();
    }

    // All other relocations reference the symbol table.
    auto SymIt = SymIndexToSym.find(Reloc.Index);
    if (SymIt == SymIndexToSym.end())
      return make_error<JITLinkError>(
          "WASM relocation references unknown symbol index " +
          Twine(Reloc.Index));

    Symbol *TargetSym = SymIt->second;
    // For FunctionIndexLEB the value to encode is the callee's function index,
    // not its executor address. Store the index in the edge addend so that
    // applyFixup can use it directly without needing a reverse address lookup.
    int64_t Addend = Reloc.Addend;
    if (EK == wasm::FunctionIndexLEB) {
      auto FIdxIt = SymToFuncIdx.find(TargetSym);
      if (FIdxIt == SymToFuncIdx.end())
        return make_error<JITLinkError>(
            "FunctionIndexLEB relocation target has no function index: " +
            Twine(*TargetSym->getName()));
      Addend = static_cast<int64_t>(FIdxIt->second);
    }
    Info->B->addEdge(EK, BlockOffset, *TargetSym, Addend);
    LLVM_DEBUG(dbgs() << "  Edge " << wasm::getEdgeKindName(EK)
                      << " at blockOffset=" << BlockOffset
                      << " → " << SymIt->second->getName() << "\n");
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
    auto *Loc = reinterpret_cast<uint8_t *>(
        B.getMutableContent(G).data() + E.getOffset());
    uint64_t Value = E.getTarget().getAddress().getValue() + E.getAddend();

    switch (E.getKind()) {
    case wasm::FunctionIndexLEB:
      // The function index is stored in the edge addend (not the target
      // address, which is a body memory address unusable as a WASM index).
      encodeULEB128(static_cast<uint64_t>(E.getAddend()), Loc, 5);
      break;
    case wasm::MemoryAddrLEB:
    case wasm::GlobalIndexLEB:
    case wasm::TypeIndexLEB:
    case wasm::TableNumberLEB:
      encodeULEB128(Value, Loc, 5);
      break;
    case wasm::TableIndexSLEB:
    case wasm::MemoryAddrSLEB:
      encodeSLEB128(static_cast<int64_t>(Value), Loc, 5);
      break;
    case wasm::MemoryAddrI32:
    case wasm::TableIndexI32:
    case wasm::FunctionIndexI32:
      support::endian::write32le(Loc, static_cast<uint32_t>(Value));
      break;
    default:
      return make_error<JITLinkError>(
          "Unsupported WASM edge kind in applyFixup: " +
          Twine(wasm::getEdgeKindName(E.getKind())));
    }
    return Error::success();
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
