multiclass WebAssemblyTableGet<WebAssemblyRegClass rc, string Name, int Opcode> {
  let mayLoad = 1, UseNamedOperandTable = 1 in
  defm "": I<(outs rc:$dst),
             (ins table_op: $table, I32:$offset),
             (outs), (ins table_op: $table),
             [], !strconcat(Name, "\t$dst, ${offset}(${table})"),
             !strconcat(Name, "\t${table}"), Opcode>;
}

defm TABLE_GET : WebAssemblyTableGet<ANYREF, "table.get", 0x25>;

def : Pat<(anyref (load (WebAssemblywrapper anyref:$addr))), (TABLE_GET anyref:$addr, (CONST_I32 0))>;
