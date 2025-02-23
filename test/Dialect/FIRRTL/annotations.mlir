// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' --split-input-file %s | FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit

// Annotations targeting the circuit work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-SAME:    annotations =
// CHECK-SAME:      {class = "circt.testNT", data = "NoTarget"}
// CHECK-SAME:      {class = "circt.test", data = "Target"}
// CHECK-SAME:      {class = "circt.test", data = "CircuitName"}
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.testNT",
    data = "NoTarget"
  },
  {
    class = "circt.test",
    data = "Target",
    target = "~Foo"
  },
  {
    class = "circt.test",
    data = "CircuitName",
    target = "Foo"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// Annotations targeting modules or external modules work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "Target",
    target = "~Foo|Foo"
  },
  {
    class = "circt.test",
    data = "ModuleName",
    target = "Foo.Foo"
  },
    {
    class = "circt.test",
    data = "ExtModule Target",
    target = "~Foo|Blackbox"
  }
]} {
  // CHECK:      firrtl.module @Foo
  // CHECK-SAME:   annotations =
  // CHECK-SAME:     {class = "circt.test", data = "Target"}
  // CHECK-SAME:     {class = "circt.test", data = "ModuleName"}
  firrtl.module @Foo() {}
  // CHECK:      firrtl.extmodule @Blackbox
  // CHECK-SAME:   annotations =
  // CHECK-SAME:     {class = "circt.test", data = "ExtModule Target"}
  firrtl.extmodule @Blackbox()
}

// -----

// Annotations targeting instances should create NLAs on the module.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
// CHECK-NEXT:    firrtl.hierpath @[[nla_c:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
// CHECK-NEXT:    firrtl.hierpath @[[nla_b:[^ ]+]] [@Foo::@[[bar_sym]],       @Bar]
// CHECK-NEXT:    firrtl.hierpath @[[nla_a:[^ ]+]] [@Foo::@[[bar_sym]],       @Bar]
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  },
  {
    class = "circt.test",
    data = "c",
    target = "~Foo|Foo/bar:Bar"
  }
]} {
  // CHECK-NEXT: firrtl.module @Bar()
  // CHECK-SAME:   annotations =
  // CHECK-SAME:     {circt.nonlocal = @[[nla_a]], class = "circt.test", data = "a"}
  // CHECK-SAME:     {circt.nonlocal = @[[nla_b]], class = "circt.test", data = "b"}
  // CHECK-SAME:     {circt.nonlocal = @[[nla_c]], class = "circt.test", data = "c"}
  firrtl.module @Bar() {}
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar sym @[[bar_sym]]
    firrtl.instance bar @Bar()
  }
}

// -----

// Test result annotations of InstanceOp.
//
// Must add inner_sym, if any subfield of a bundle type has nonlocal anchor.
// Otherwise, the nla will be illegal, without any inner_sym.
// Test on port and wire.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
// CHECK-NEXT:    firrtl.hierpath @[[nla_4:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
// CHECK-NEXT:    firrtl.hierpath @[[nla_3:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
// CHECK-NEXT:    firrtl.hierpath @[[nla_2:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
// CHECK-NEXT:    firrtl.hierpath @[[nla_1:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
// CHECK-NEXT:    firrtl.hierpath @[[nla_0:[^ ]+]] [@Foo::@[[bar_sym:[^ ]+]], @Bar]
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = 0,
    target = "~Foo|Foo>bar.a"
  },
  {
    class = "circt.test",
    data = 1,
    target = "~Foo|Foo>bar.b.baz"
  },
  {
    class = "circt.test",
    data = 2,
    target = "~Foo|Foo/bar:Bar>b.qux"
  },
  {
    class = "circt.test",
    data = 3,
    target = "~Foo|Foo/bar:Bar>d.qux"
  },
  {
    class = "circt.test",
    data = 4,
    target = "Foo.Foo.bar.c"
  }
]} {
  // CHECK-NEXT: firrtl.module @Bar
  // CHECK-SAME:   in %a
  // CHECK-SAME:     {circt.nonlocal = @[[nla_0]], class = "circt.test", data = 0 : i64}
  // CHECK-SAME:   out %b
  // CHECK-SAME:     {circt.fieldID = 1 : i32, circt.nonlocal = @[[nla_1]], class = "circt.test", data = 1 : i64}
  // CHECK-SAME:     {circt.fieldID = 2 : i32, circt.nonlocal = @[[nla_2]], class = "circt.test", data = 2 : i64}
  // CHECK-SAME:   out %c
  // CHECK-SAME:     {circt.nonlocal = @[[nla_4]], class = "circt.test", data = 4 : i64}
  firrtl.module @Bar(
    in %a: !firrtl.uint<1>,
    out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>,
    out %c: !firrtl.uint<1>
  ) {
    // CHECK-NEXT: %d = firrtl.wire
    // CHECK-NOT:    sym
    // CHECK-SAME:   {circt.fieldID = 2 : i32, circt.nonlocal = @[[nla_3]], class = "circt.test", data = 3 : i64}
    %d = firrtl.wire : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar sym @[[bar_sym]]
    %bar_a, %bar_b, %bar_c = firrtl.instance bar @Bar(
      in a: !firrtl.uint<1>,
      out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>,
      out c: !firrtl.uint<1>
    )
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a Foo should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: chirrtl.combmem
    // CHECK-SAME:   {class = "circt.test", data = "a"}
    // CHECK-SAME:   {class = "circt.test", data = "b"}
    %bar = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 8>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a memory should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.mem
    // CHECK-SAME:   {class = "circt.test", data = "a"}
    // CHECK-SAME:   {class = "circt.test", data = "b"}
    %bar_r = firrtl.mem Undefined {
       depth = 16 : i64,
       name = "bar",
       portNames = ["r"],
       readLatency = 0 : i32,
       writeLatency = 1 : i32
     } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

// Test result annotations of MemOp.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-NOT:     rawAnnotations
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar.r"
  }
  ,
  {
    class = "circt.test",
    data = "b",
    target = "~Foo|Foo>bar.r.data.baz"
  }
  ,
  {
    class = "circt.test",
    data = "c",
    target = "~Foo|Foo>bar.w.en"
  }
  ,
  {
    class = "circt.test",
    data = "d",
    target = "~Foo|Foo>bar.w.data.qux"
  }
]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.mem
    // CHECK-SAME:   portAnnotations =
    // CHECK-SAME:     [{class = "circt.test", data = "a"}, {circt.fieldID = 5 : i32, class = "circt.test", data = "b"}]
    // CHECK-SAME:     [{circt.fieldID = 2 : i32, class = "circt.test", data = "c"}, {circt.fieldID = 6 : i32, class = "circt.test", data = "d"}]
    %bar_r, %bar_w = firrtl.mem interesting_name Undefined {
      depth = 16 : i64,
      name = "bar",
      portNames = ["r", "w"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a node should work.  This
// shouldn't crash if the node is in a nested block.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.baz"
  }
]} {
  firrtl.module @Foo(
    in %clock: !firrtl.clock,
    in %cond_0: !firrtl.uint<1>,
    in %cond_1: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %bar = firrtl.node %c0_ui1  : !firrtl.uint<1>
    firrtl.when %cond_0 {
      firrtl.when %cond_1 {
        %baz = firrtl.node %c0_ui1  : !firrtl.uint<1>
      }
    }
  }
}

// CHECK:      firrtl.module @Foo
// CHECK:        %bar = firrtl.node
// CHECK-SAME:     annotations = [{class = "circt.test", data = "a"}
// CHECK:        %baz = firrtl.node
// CHECK-SAME:     annotations = [{class = "circt.test", data = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at a wire should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  firrtl.module @Foo() {
    %bar = firrtl.wire : !firrtl.uint<1>
  }
}

// CHECK:      %bar = firrtl.wire
// CHECK-SAME:   annotations = [{class = "circt.test", data = "a"}, {class = "circt.test", data = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at a register should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.baz"
  }
]} {
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bar = firrtl.reg %clock  : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %baz = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK:      %bar = firrtl.reg
// CHECK-SAME:   annotations = [{class = "circt.test", data = "a"}]
// CHECK:      %baz = firrtl.regreset
// CHECK-SAME:   annotations = [{class = "circt.test", data = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at an SeqMem should work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.bar"
  }
]} {
  firrtl.module @Foo() {
    %bar = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<1>, 8>
  }
}

// CHECK:      chirrtl.seqmem
// CHECK-SAME:   annotations = [{class = "circt.test", data = "a"}, {class = "circt.test", data = "b"}]

// -----

// Subfield/Subindex annotations should be parsed correctly on wires
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "one",
    target = "~Foo|Foo>bar[0]"
  },
  {
    class = "circt.test",
    data = "two",
    target = "~Foo|Foo>bar[1].baz"
  }
]} {
  firrtl.module @Foo() {
    %bar = firrtl.wire : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// CHECK:      %bar = firrtl.wire {annotations =
// CHECK-SAME:   {circt.fieldID = 1 : i32, class = "circt.test", data = "one"}
// CHECK-SAME:   {circt.fieldID = 5 : i32, class = "circt.test", data = "two"}

// -----

// Subfield/Subindex annotations should be parsed correctly on registers
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "one",
    target = "~Foo|Foo>bar[0]"
  },
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[1].baz",
    data = "two"
  }
]} {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// CHECK:      %bar = firrtl.reg %clock {annotations =
// CHECK-SAME:   {circt.fieldID = 1 : i32, class = "circt.test", data = "one"}
// CHECK-SAME:   {circt.fieldID = 5 : i32, class = "circt.test", data = "two"}

// -----

// Subindices should not get sign-extended and cause problems.  This circuit has
// caused bugs in the past.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Foo>w[9]"
  }
]} {
  firrtl.module @Foo() {
    %w = firrtl.wire  : !firrtl.vector<uint<1>, 18>
  }
}

// CHECK:      %w = firrtl.wire {annotations =
// CHECK-SAME:   {circt.fieldID = 10 : i32, class = "circt.test", data = "a"}

// -----

// A ReferenceTarget/ComponentName pointing at a module/extmodule port should
// work.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    data = "a",
    target = "~Foo|Bar>bar"
  },
  {
    class = "circt.test",
    data = "b",
    target = "Foo.Foo.foo"
  }
]} {
  firrtl.extmodule @Bar(in bar: !firrtl.uint<1>)
  firrtl.module @Foo(in %foo: !firrtl.uint<1>) {
    %bar_bar = firrtl.instance bar  @Bar(in bar: !firrtl.uint<1>)
    firrtl.strictconnect %bar_bar, %foo : !firrtl.uint<1>
  }
}

// CHECK:      firrtl.extmodule @Bar
// CHECK-SAME:   [[_:.+]] [{class = "circt.test", data = "a"}]
// CHECK:      firrtl.module @Foo
// CHECK-SAME:   %foo: [[_:.+]] [{class = "circt.test", data = "b"}]

// -----

// A module with an instance in its body which has the same name as the module
// itself should not cause issues attaching annotations.
// https://github.com/llvm/circt/issues/2709
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo/Foo:Example"
  }
]} {
  firrtl.module @Example() {}
  firrtl.module @Foo() {
    firrtl.instance Foo @Example()
  }
}

// CHECK-LABEL:  firrtl.circuit "Foo"
// CHECK:          firrtl.hierpath @[[nla:[^ ]+]] [@Foo::@Foo, @Example]
// CHECK:          firrtl.module @Example() attributes {
// CHECK-SAME:       annotations = [{circt.nonlocal = @[[nla]], class = "circt.test"}]
// CHECK:          firrtl.module @Foo()
// CHECK:            firrtl.instance Foo sym @Foo @Example()

// -----

// Multiple non-local Annotations are supported.
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {class = "circt.test", data = "a", target = "~Foo|Foo/bar:Bar/baz:Baz"},
  {class = "circt.test", data = "b", target = "~Foo|Foo/bar:Bar/baz:Baz"}
]} {
  firrtl.module @Baz() {}
  firrtl.module @Bar() {
    firrtl.instance baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
}
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK:         firrtl.hierpath @[[nla_b:[^ ]+]] [@Foo::@bar, @Bar::@baz, @Baz]
// CHECK:         firrtl.hierpath @[[nla_a:[^ ]+]] [@Foo::@bar, @Bar::@baz, @Baz]
// CHECK:         firrtl.module @Baz
// CHECK-SAME:      annotations = [{circt.nonlocal = @[[nla_a]], class = "circt.test", data = "a"}, {circt.nonlocal = @[[nla_b]], class = "circt.test", data = "b"}]
// CHECK:         firrtl.module @Bar()
// CHECK:           firrtl.instance baz sym @baz @Baz()
// CHECK:           firrtl.module @Foo()
// CHECK:           firrtl.instance bar sym @bar @Bar()

// -----

firrtl.circuit "memportAnno"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~memportAnno|memportAnno/foo:Foo>memory.w"
  }
]} {
  firrtl.module @memportAnno() {
    firrtl.instance foo @Foo()
  }
  firrtl.module @Foo() {
    %memory_w = firrtl.mem Undefined {
      depth = 16 : i64,
      name = "memory",
      portNames = ["w"],
      readLatency = 1 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// CHECK-LABEL: firrtl.circuit "memportAnno"  {
// CHECK:        firrtl.hierpath @nla [@memportAnno::@foo, @Foo]
// CHECK:        %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portAnnotations
// CHECK-SAME:   [{circt.nonlocal = @nla, class = "circt.test"}]

// -----

// Test annotation targeting an instance port
// https://github.com/llvm/circt/issues/3340
firrtl.circuit "instportAnno" attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~instportAnno|instportAnno/bar:Bar>baz.a"
  }
]} {
  firrtl.module @Baz(out %a: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %a, %invalid_ui1 : !firrtl.uint<1>
  }
  firrtl.module @Bar() {
    %baz_a = firrtl.instance baz @Baz(out a: !firrtl.uint<1>)
  }
  firrtl.module @instportAnno() {
    firrtl.instance bar @Bar()
  }
}

// CHECK-LABEL: firrtl.circuit "instportAnno"
// CHECK:        firrtl.hierpath @[[HIER:[^ ]+]] [@instportAnno::@bar, @Bar::@baz, @Baz]
// CHECK:        firrtl.module @Baz
// CHECK-SAME:     {circt.nonlocal = @[[HIER]], class = "circt.test"}

// -----

// CHECK-LABEL: firrtl.circuit "Aggregates"
firrtl.circuit "Aggregates" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Aggregates|Aggregates>vector[1][1][1]"},
  {class = "circt.test", target = "~Aggregates|Aggregates>bundle.a.b.c"}
  ]} {
  firrtl.module @Aggregates() {
    // CHECK: {annotations = [{circt.fieldID = 14 : i32, class = "circt.test"}]}
    %vector = firrtl.wire  : !firrtl.vector<vector<vector<uint<1>, 2>, 2>, 2>
    // CHECK: {annotations = [{circt.fieldID = 3 : i32, class = "circt.test"}]}
    %bundle = firrtl.wire  : !firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>
  }
}

// -----

// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: firrtl.hierpath @nla_1 [@FooNL::@baz, @BazNL::@bar, @BarNL]
// CHECK: firrtl.hierpath @nla_0 [@FooNL::@baz, @BazNL::@bar, @BarNL]
// CHECK: firrtl.hierpath @nla [@FooNL::@baz, @BazNL::@bar, @BarNL]
// CHECK: firrtl.module @BarNL
// CHECK: %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire sym @w2 {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla_1, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>>
// CHECK: firrtl.instance bar sym @bar @BarNL()
// CHECK: firrtl.instance baz sym @baz @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "FooNL"  attributes {rawAnnotations = [
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL"},
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"},
  {class = "circt.test", nl = "nl2", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w2.b[2]"},
  {class = "circt.test", nl = "nl3", target = "~FooNL|FooL>w3"}
  ]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  sym @w : !firrtl.uint
    %w2 = firrtl.wire sym @w2 : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance bar sym @bar @BarNL()
  }
  firrtl.module @FooNL() {
    firrtl.instance baz sym @baz @BazNL()
  }
  firrtl.module @FooL() {
    %w3 = firrtl.wire: !firrtl.uint
  }
}

// -----

// Non-local annotations on memory ports should work.

// CHECK-LABEL: firrtl.circuit "MemPortsNL"
// CHECK: firrtl.hierpath @nla [@MemPortsNL::@child, @Child]
// CHECK: firrtl.module @Child()
// CHECK:   %bar_r = firrtl.mem
// CHECK-NOT: sym
// CHECK-SAME: portAnnotations = {{\[}}[{circt.nonlocal = @nla, class = "circt.test", nl = "nl"}]]
// CHECK: firrtl.module @MemPortsNL()
// CHECK:   firrtl.instance child sym @child
firrtl.circuit "MemPortsNL" attributes {rawAnnotations = [
  {class = "circt.test", nl = "nl", target = "~MemPortsNL|MemPortsNL/child:Child>bar.r"}
  ]}  {
  firrtl.module @Child() {
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  firrtl.module @MemPortsNL() {
    firrtl.instance child @Child()
  }
}

// -----

// Annotations on ports should work.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|PortTest>in"}
  ]} {
  firrtl.module @PortTest(in %in : !firrtl.uint<1>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.uint<1>)
  }
}

// -----

// Subannotations on ports should work.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|PortTest>in.a"}
  ]} {
  // CHECK: firrtl.module @PortTest(in %in: !firrtl.bundle<a: uint<1>> [{circt.fieldID = 1 : i32, class = "circt.test"}])
  firrtl.module @PortTest(in %in : !firrtl.bundle<a: uint<1>>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.bundle<a: uint<1>>)
  }
}
// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|Test>exttest"}
  ]} {
  // CHECK: firrtl.hierpath @nla [@Test::@exttest, @ExtTest]
  // CHECK: firrtl.extmodule @ExtTest() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]}
  firrtl.extmodule @ExtTest()

  firrtl.module @Test() {
    // CHECK: firrtl.instance exttest sym @exttest @ExtTest()
    firrtl.instance exttest @ExtTest()
  }
}

// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|Test>exttest.in"}
  ]} {
  // CHECK: firrtl.hierpath @nla [@Test::@exttest, @ExtTest]
  // CHECK: firrtl.extmodule @ExtTest(in in: !firrtl.uint<1> [{circt.nonlocal = @nla, class = "circt.test"}])
  firrtl.extmodule @ExtTest(in in: !firrtl.uint<1>)

  firrtl.module @Test() {
    // CHECK: %exttest_in = firrtl.instance exttest sym @exttest @ExtTest(in in: !firrtl.uint<1>)
    firrtl.instance exttest @ExtTest(in in : !firrtl.uint<1>)
  }
}

// -----

// DontTouchAnnotations are placed on the things they target.

firrtl.circuit "Foo"  attributes {
  rawAnnotations = [
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_0"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_1"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_2"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_3"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_4"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_5"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_6"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_8"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_9.a"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo/bar:Bar>_T.a"}]} {
  // CHECK:      firrtl.hierpath @nla [@Foo::@bar, @Bar]
  // CHECK-NEXT: firrtl.module @Foo
  firrtl.module @Foo(in %reset: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    // CHECK-NEXT: %_T_0 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_0 = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT: %_T_1 = firrtl.node %_T_0 {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_1 = firrtl.node %_T_0  : !firrtl.uint<1>
    // CHECK-NEXT: %_T_2 = firrtl.reg %clock {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_2 = firrtl.reg %clock  : !firrtl.uint<1>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    // CHECK: %_T_3 = firrtl.regreset
    // CHECK-SAME: {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
    // CHECK-NEXT: %_T_4 = chirrtl.seqmem
    // CHECK-SAME: {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_4 = chirrtl.seqmem Undefined  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK-NEXT: %_T_5 = chirrtl.combmem {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_5 = chirrtl.combmem  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: chirrtl.memoryport Infer %_T_5 {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_6_data, %_T_6_port = chirrtl.memoryport Infer %_T_5  {name = "_T_6"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %_T_6_port[%reset], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
    // CHECK: firrtl.mem
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_8_w = firrtl.mem Undefined  {depth = 8 : i64, name = "_T_8", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %aggregate = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %_T_9 = firrtl.node %aggregate {annotations = [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_9 = firrtl.node %aggregate  : !firrtl.bundle<a: uint<1>>
    firrtl.instance bar @Bar()
  }
  firrtl.module @Bar() {
    //  CHECK: %_T = firrtl.wire {annotations = [{circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T = firrtl.wire : !firrtl.bundle<a: uint<1>>
  }
}

// -----

firrtl.circuit "GCTInterface"  attributes {annotations = [{unrelatedAnnotation}], rawAnnotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", companion = "~GCTInterface|view_companion", name = "view", parent = "~GCTInterface|GCTInterface", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "ViewName", elements = [{description = "the register in GCTInterface", name = "register", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Register", elements = [{name = "_2", tpe = {class = "sifive.enterprise.grandcentral.AugmentedVectorType", elements = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 0 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 1 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}]}}, {name = "_0_inst", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "_0_def", elements = [{name = "_1", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_1"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}, {name = "_0", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_0"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}]}}, {description = "the port 'a' in GCTInterface", name = "port", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [], module = "GCTInterface", path = [], ref = "a"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}]} {
  firrtl.module private @view_companion() {
    firrtl.skip
  }
  firrtl.module @GCTInterface(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
    firrtl.instance view_companion  @view_companion()
  }
}

// CHECK-LABEL: firrtl.circuit "GCTInterface"

// The interface definition should show up as a circuit annotation.  Nested
// interfaces show up as nested bundle types and not as separate interfaces.
// CHECK-SAME: annotations
// CHECK-SAME: {unrelatedAnnotation}
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:  defName = "ViewName",
// CHECK-SAME:  elements = [
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:     defName = "Register",
// CHECK-SAME:     description = "the register in GCTInterface",
// CHECK-SAME:     elements = [
// CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
// CHECK-SAME:        elements = [
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_2_0:[0-9]+]] : i64,
// CHECK-SAME:           name = "_2"},
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_2_1:[0-9]+]] : i64,
// CHECK-SAME:           name = "_2"}],
// CHECK-SAME:        name = "_2"},
// CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:        defName = "_0_def",
// CHECK-SAME:        elements = [
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_1:[0-9]+]] : i64,
// CHECK-SAME:           name = "_1"},
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_0:[0-9]+]] : i64,
// CHECK-SAME:           name = "_0"}],
// CHECK-SAME:        name = "_0_inst"}],
// CHECK-SAME:     name = "register"},
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:     description = "the port 'a' in GCTInterface",
// CHECK-SAME:     id = [[ID_port:[0-9]+]] : i64,
// CHECK-SAME:     name = "port"}],
// CHECK-SAME:  id = [[ID_ViewName:[0-9]+]] : i64,
// CHECK-SAME:  name = "view"}

// The companion should be marked.
// CHECK: firrtl.module private @view_companion
// CHECK-SAME: annotations
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
// CHECK-SAME:  id = [[ID_ViewName]] : i64,
// CHECK-SAME:  type = "companion"}

// The parent should be annotated. Additionally, this example has all the
// members of the interface inside the parent.  Both port "a" and register
// "r" should be annotated.
// CHECK: firrtl.module @GCTInterface
// CHECK-SAME: %a: !firrtl.uint<1> [
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    d = [[ID_port]] : i64}
// CHECK-SAME: annotations = [
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
// CHECK-SAME:    id = [[ID_ViewName]] : i64,
// CHECK-SAME:    name = "view",
// CHECK-SAME:    type = "parent"}]
// CHECK: firrtl.reg
// CHECK-NOT:  sym
// CHECK-SAME: annotations
// CHECK-SAME:   {circt.fieldID = 2 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_0]] : i64}
// CHECK-SAME:   {circt.fieldID = 3 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_1]] : i64}
// CHECK-SAME:   {circt.fieldID = 6 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_2_1]] : i64}
// CHECK-SAME:   {circt.fieldID = 5 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_2_0]] : i64}

// -----

firrtl.circuit "Foo"  attributes {rawAnnotations = [{class = "sifive.enterprise.grandcentral.ViewAnnotation", companion = "~Foo|Bar_companion", name = "Bar", parent = "~Foo|Foo", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "View", elements = [{description = "a string", name = "string", tpe = {class = "sifive.enterprise.grandcentral.AugmentedStringType", value = "hello"}}, {description = "a boolean", name = "boolean", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBooleanType", value = false}}, {description = "an integer", name = "integer", tpe = {class = "sifive.enterprise.grandcentral.AugmentedIntegerType", value = 42 : i64}}, {description = "a double", name = "double", tpe = {class = "sifive.enterprise.grandcentral.AugmentedDoubleType", value = 3.140000e+00 : f64}}]}}]} {
  firrtl.extmodule private @Bar_companion()
  firrtl.module @Foo() {
     firrtl.instance Bar_companion @Bar_companion()
   }
}

// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-SAME: annotations = [{class = "[[_:.+]]AugmentedBundleType", [[_:.+]] elements = [{
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedStringType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedBooleanType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedIntegerType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedDoubleType"

// -----

// SiFive-custom annotations related to the GrandCentral utility.  These
// annotations do not conform to standard SingleTarget or NoTarget format and
// need to be manually split up.

// Test sifive.enterprise.grandcentral.DataTapsAnnotation with all possible
// variants of DataTapKeys.

firrtl.circuit "GCTDataTap" attributes {rawAnnotations = [{
  blackBox = "~GCTDataTap|DataTap",
  class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
  keys = [
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_0",
      source = "~GCTDataTap|GCTDataTap>r"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_1[0]",
      source = "~GCTDataTap|GCTDataTap>r"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_2",
      source = "~GCTDataTap|GCTDataTap>w.a"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_3[0]",
      source = "~GCTDataTap|GCTDataTap>w.a"
    },
    {
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "baz.qux",
      module = "~GCTDataTap|BlackBox",
      portName = "~GCTDataTap|DataTap>_4"
    },
    {
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "baz.quz",
      module = "~GCTDataTap|BlackBox",
      portName = "~GCTDataTap|DataTap>_5[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
      portName = "~GCTDataTap|DataTap>_6"
    },
    {
      class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
      portName = "~GCTDataTap|DataTap>_7[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
      literal = "UInt<16>(\22h2a\22)",
      portName = "~GCTDataTap|DataTap>_8"
    },
    {
      class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
      literal = "UInt<16>(\22h2a\22)",
      portName = "~GCTDataTap|DataTap>_9[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_10",
      source = "~GCTDataTap|GCTDataTap/im:InnerMod>w"
    }
  ]
}]} {
  firrtl.extmodule private @DataTap(
    out _0: !firrtl.uint<1>,
    out _1: !firrtl.vector<uint<1>, 1>,
    out _2: !firrtl.uint<1>,
    out _3: !firrtl.vector<uint<1>, 1>,
    out _4: !firrtl.uint<1>,
    out _5: !firrtl.vector<uint<1>, 1>,
    out _6: !firrtl.uint<1>,
    out _7: !firrtl.vector<uint<1>, 1>,
    out _8: !firrtl.uint<1>,
    out _9: !firrtl.vector<uint<1>, 1>,
    out _10: !firrtl.uint<1>
  ) attributes {defname = "DataTap"}
  firrtl.extmodule private @BlackBox() attributes {defname = "BlackBox"}
  firrtl.module private @InnerMod() {
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @GCTDataTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %DataTap__0, %DataTap__1, %DataTap__2, %DataTap__3, %DataTap__4, %DataTap__5, %DataTap__6, %DataTap__7, %DataTap__8, %DataTap__9, %DataTap__10 = firrtl.instance DataTap  @DataTap(out _0: !firrtl.uint<1>, out _1: !firrtl.vector<uint<1>, 1>, out _2: !firrtl.uint<1>, out _3: !firrtl.vector<uint<1>, 1>, out _4: !firrtl.uint<1>, out _5: !firrtl.vector<uint<1>, 1>, out _6: !firrtl.uint<1>, out _7: !firrtl.vector<uint<1>, 1>, out _8: !firrtl.uint<1>, out _9: !firrtl.vector<uint<1>, 1>, out _10: !firrtl.uint<1>)
    firrtl.instance BlackBox @BlackBox()
    firrtl.instance im @InnerMod()
  }
}

// CHECK-LABEL: firrtl.circuit "GCTDataTap"
// CHECK:      firrtl.hierpath [[NLA:@.+]] [@GCTDataTap::@im, @InnerMod]

// CHECK-LABEL: firrtl.extmodule private @DataTap

// CHECK-SAME: _0: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID:[0-9]+]] : i64
// CHECK-SAME:   portID = [[PORT_ID_0:[0-9]+]] : i64

// CHECK-SAME: _1: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_1:[0-9]+]] : i64

// CHECK-SAME: _2: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_2:[0-9]+]] : i64

// CHECK-SAME: _3: !firrtl.vector<uint<1>, 1> [
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_3:[0-9]+]] : i64

// CHECK-SAME: _4: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_4:[0-9]+]] : i64

// CHECK-SAME: _5: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_5:[0-9]+]] : i64

// CHECK-SAME: _6: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DeletedDataTapKey"
// CHECK-SAME:   id = [[ID]] : i64

// CHECK-SAME: _7: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DeletedDataTapKey"
// CHECK-SAME:   id = [[ID]] : i64

// CHECK-SAME: _8: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.LiteralDataTapKey"
// CHECK-SAME:   literal = "UInt<16>(\22h2a\22)"

// CHECK-SAME: _9: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.LiteralDataTapKey"
// CHECK-SAME:   literal = "UInt<16>(\22h2a\22)"

// CHECK-SAME: _10: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_6:[0-9]+]] : i64

// CHECK-SAME: annotations = [
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}
// CHECK-SAME: ]

// CHECK-LABEL: firrtl.extmodule private @BlackBox
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source",
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     internalPath = "baz.quz",
// CHECK-SAME:     portID = [[PORT_ID_5]] : i64
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source",
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     internalPath = "baz.qux",
// CHECK-SAME:     portID = [[PORT_ID_4]] : i64
// CHECK-SAME:   }
// CHECK-SAME: ]

// CHECK-LABEL: firrtl.module private @InnerMod
// CHECK-NEXT: %w = firrtl.wire
// CHECK-NOT:  sym
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     circt.nonlocal = [[NLA]]
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = [[PORT_ID_6]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// CHECK: firrtl.module @GCTDataTap
// CHECK-LABEL: firrtl.reg
// CHECK-NOT:  sym
// CHECk-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = [[PORT_ID_0]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// CHECK-LABEL: firrtl.wire
// CHECK-NOT:  sym
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 1
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = [[PORT_ID_3]]
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
// CHECK-SAME:      id = [[ID]]
// CHECK-SAME:      portID = [[PORT_ID_2]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// -----

// Test sifive.enterprise.grandcentral.MemTapAnnotation
firrtl.circuit "GCTMemTap" attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.MemTapAnnotation",
  source = "~GCTMemTap|GCTMemTap>mem",
  taps = ["GCTMemTap.MemTap.mem[0]", "GCTMemTap.MemTap.mem[1]"]
}]} {
  firrtl.extmodule private @MemTap(out mem: !firrtl.vector<uint<1>, 2>) attributes {defname = "MemTap"}
  firrtl.module @GCTMemTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %mem = chirrtl.combmem  : !chirrtl.cmemory<uint<1>, 2>
    %MemTap_mem = firrtl.instance MemTap  @MemTap(out mem: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %MemTap_mem[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %MemTap_mem[0] : !firrtl.vector<uint<1>, 2>
    %memTap = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %memTap[1] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %memTap[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
    firrtl.strictconnect %2, %0 : !firrtl.uint<1>
  }
}


// CHECK-LABEL: firrtl.circuit "GCTMemTap"

// CHECK-LABEL: firrtl.extmodule private @MemTap
// CHECK-SAME: mem: !firrtl.vector<uint<1>, 2> [
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 2
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.MemTapAnnotation.port"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = 1
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 1
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.MemTapAnnotation.port"
// CHECK-SAME:     id = [[ID:[0-9]+]] : i64
// CHECK-SAME:     portID = 0
// CHECK-SAME:   }

// CHECK-LABEL: firrtl.module @GCTMemTap
// CHECK: %mem = chirrtl.combmem
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.MemTapAnnotation.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// -----

firrtl.circuit "Sub"  attributes {
  rawAnnotations = [
    {
      annotations = [],
      circuit = "",
      circuitPackage = "other",
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
      sinkTargets = [
        {_1 = "~Top|Foo>clock", _2 = "~Sub|Sub>clockSink"},
        {_1 = "~Top|Foo>dataIn.a.b.c", _2 = "~Sub|Sub>dataSink.u"},
        {_1 = "~Top|Foo>dataIn.d", _2 = "~Sub|Sub>dataSink.v"},
        {_1 = "~Top|Foo>dataIn.e", _2 = "~Sub|Sub>dataSink.w"}
      ],
      sourceTargets = [
        {_1 = "~Top|Top>clock", _2 = "~Sub|Sub>clockSource"},
        {_1 = "~Top|Foo>dataOut.x.y.z", _2 = "~Sub|Sub>dataSource.u"},
        {_1 = "~Top|Foo>dataOut.w", _2 = "~Sub|Sub>dataSource.v"},
        {_1 = "~Top|Foo>dataOut.p", _2 = "~Sub|Sub>dataSource.w"}
      ]
    }
  ]
} {
  firrtl.extmodule private @SubExtern(
    in clockIn: !firrtl.clock,
    out clockOut: !firrtl.clock,
    in someInput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>, out someOutput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
  )
  firrtl.module @Sub() {
    %clockSource = firrtl.wire interesting_name  : !firrtl.clock
    %clockSink = firrtl.wire interesting_name  : !firrtl.clock
    %dataSource = firrtl.wire interesting_name  : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
    %dataSink = firrtl.wire interesting_name  : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
    %ext_clockIn, %ext_clockOut, %ext_someInput, %ext_someOutput = firrtl.instance ext interesting_name  @SubExtern(in clockIn: !firrtl.clock, out clockOut: !firrtl.clock, in someInput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>, out someOutput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>)
    firrtl.strictconnect %ext_clockIn, %clockSource : !firrtl.clock
    firrtl.strictconnect %ext_someInput, %dataSource : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
    firrtl.strictconnect %clockSink, %ext_clockOut : !firrtl.clock
    firrtl.strictconnect %dataSink, %ext_someOutput : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
  }
}

// CHECK-LABEL: firrtl.circuit "Sub"
// CHECK-SAME:    {annotations = [], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = [[id:[0-9]+]] : i64, isSubCircuit = true}
//
// CHECK:         firrtl.module @Sub()
// CHECK-SAME:      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = [[id]] : i64}
// CHECK-NEXT:      %clockSource = firrtl.wire
// CHECK-SAME:        {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = [[id]] : i64, peer = "~Top|Top>clock", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-NEXT:      %clockSink = firrtl.wire
// CHECK-SAME:        {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = [[id]] : i64, peer = "~Top|Foo>clock", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-NEXT:      %dataSource = firrtl.wire
// CHECK-SAME:        {circt.fieldID = 3 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = [[id]] : i64, peer = "~Top|Foo>dataOut.p", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {circt.fieldID = 3 : i32, class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-SAME:        {circt.fieldID = 2 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = [[id]] : i64, peer = "~Top|Foo>dataOut.w", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-SAME:        {circt.fieldID = 1 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = [[id]] : i64, peer = "~Top|Foo>dataOut.x.y.z", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-NEXT:      %dataSink = firrtl.wire
// CHECK-SAME:        {circt.fieldID = 3 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = [[id]] : i64, peer = "~Top|Foo>dataIn.e", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {circt.fieldID = 3 : i32, class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-SAME:        {circt.fieldID = 2 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = [[id]] : i64, peer = "~Top|Foo>dataIn.d", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}
// CHECK-SAME:        {circt.fieldID = 1 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = [[id]] : i64, peer = "~Top|Foo>dataIn.a.b.c", side = "local", targetId = {{[0-9]+}} : i64}
// CHECK-SAME:        {circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}
