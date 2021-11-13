# RUN: %PYTHON% %s %t
# RUN: FileCheck %s --input-file %t/WireNames.sv --check-prefix=OUTPUT

from pycde import (Output, Input, module, generator, types, dim, System)

import sys


@module
class WireNames:
  clk = Input(types.i1)
  sel = Input(types.i2)
  data_in = Input(dim(32, 3))

  a = Output(types.i32)
  b = Output(types.i32)

  @generator
  def build(ports):
    foo = ports.data_in[0]
    foo.name = "foo"
    arr_data = dim(32, 4).create([1, 2, 3, 4], "arr_data")
    ports.set_all_ports({
        'a': foo.reg(ports.clk).reg(ports.clk),
        'b': arr_data[ports.sel],
    })


wiresys = System([WireNames], output_directory=sys.argv[1])
wiresys.generate()
wiresys.run_passes()
wiresys.print()
wiresys.emit_outputs()

# OUTPUT-LABEL: module WireNames
# OUTPUT: reg [31:0] foo__reg1;
# OUTPUT: reg [31:0] foo__reg2;
# OUTPUT: wire [3:0][31:0] array_data = {{{{}}32'h4}, {32'h3}, {32'h2}, {32'h1}};
# OUTPUT: always_ff @(posedge clk) begin
# OUTPUT:   foo__reg1 <= data_in[2'h0];
# OUTPUT:   foo__reg2 <= foo__reg1;
# OUTPUT: end
# OUTPUT: assign a = foo__reg2;
# OUTPUT: assign b = array_data[sel];
