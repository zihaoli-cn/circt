//===- RTL/Ops.h - Declare RTL dialect operations ---------------*- C++ -*-===//
//
// This file declares the operation classes for the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRT_DIALECT_RTL_OPS_H
#define CIRT_DIALECT_RTL_OPS_H

#include "cirt/Dialect/RTL/Dialect.h"
//#include "cirt/Dialect/FIRRTL/Types.h"
//#include "mlir/IR/Builders.h"
//#include "mlir/IR/FunctionSupport.h"
//#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffects.h"

namespace cirt {
namespace rtl {

#define GET_OP_CLASSES
#include "cirt/Dialect/RTL/RTL.h.inc"

} // namespace rtl
} // namespace cirt

#endif // CIRT_DIALECT_RTL_OPS_H
