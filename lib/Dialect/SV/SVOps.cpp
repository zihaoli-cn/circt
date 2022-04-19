//===- SVOps.cpp - Implement the SV operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the SV ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace sv;

/// Return true if the specified operation is an expression.
bool sv::isExpression(Operation *op) {
  return isa<VerbatimExprOp, VerbatimExprSEOp, GetModportOp,
             ReadInterfaceSignalOp, ConstantXOp, ConstantZOp>(op);
}

LogicalResult sv::verifyInProceduralRegion(Operation *op) {
  if (op->getParentOp()->hasTrait<sv::ProceduralRegion>())
    return success();
  op->emitError() << op->getName() << " should be in a procedural region";
  return failure();
}

LogicalResult sv::verifyInNonProceduralRegion(Operation *op) {
  if (!op->getParentOp()->hasTrait<sv::ProceduralRegion>())
    return success();
  op->emitError() << op->getName() << " should be in a non-procedural region";
  return failure();
}

/// Returns the operation registered with the given symbol name with the regions
/// of 'symbolTableOp'. recurse through nested regions which don't contain the
/// symboltable trait. Returns nullptr if no valid symbol was found.
static Operation *lookupSymbolInNested(Operation *symbolTableOp,
                                       StringRef symbol) {
  Region &region = symbolTableOp->getRegion(0);
  if (region.empty())
    return nullptr;

  // Look for a symbol with the given name.
  StringAttr symbolNameId = StringAttr::get(symbolTableOp->getContext(),
                                            SymbolTable::getSymbolAttrName());
  for (Block &block : region)
    for (Operation &nestedOp : block) {
      auto nameAttr = nestedOp.getAttrOfType<StringAttr>(symbolNameId);
      if (nameAttr && nameAttr.getValue() == symbol)
        return &nestedOp;
      if (!nestedOp.hasTrait<OpTrait::SymbolTable>() &&
          nestedOp.getNumRegions()) {
        if (auto *nop = lookupSymbolInNested(&nestedOp, symbol))
          return nop;
      }
    }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        NamedAttrList &resultAttrs) {

  if (parser.parseOptionalAttrDict(resultAttrs))
    return failure();

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  bool hadName = llvm::any_of(resultAttrs, [](NamedAttribute attr) {
    return attr.getName() == "name";
  });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadName)
    return success();

  // If there is no explicit name attribute, get it from the SSA result name.
  // If numeric, just use an empty name.
  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  resultAttrs.push_back({StringAttr::get(context, "name"), nameAttr});
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 DictionaryAttr attr) {
  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  bool namesDisagree = false;

  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto expectedName = op->getAttrOfType<StringAttr>("name").getValue();
  auto actualName = tmpStream.str().drop_front();
  if (actualName != expectedName) {
    // Anonymous names are printed as digits, which is fine.
    if (!expectedName.empty() || !isdigit(actualName[0]))
      namesDisagree = true;
  }

  if (namesDisagree)
    p.printOptionalAttrDict(op->getAttrs(),
                            {SymbolTable::getSymbolAttrName(),
                             hw::InnerName::getInnerNameAttrName()});
  else
    p.printOptionalAttrDict(op->getAttrs(),
                            {"name", SymbolTable::getSymbolAttrName(),
                             hw::InnerName::getInnerNameAttrName()});
}

//===----------------------------------------------------------------------===//
// VerbatimExprOp
//===----------------------------------------------------------------------===//

/// Get the asm name for sv.verbatim.expr and sv.verbatim.expr.se.
static void
getVerbatimExprAsmResultNames(Operation *op,
                              function_ref<void(Value, StringRef)> setNameFn) {
  // If the string is macro like, then use a pretty name.  We only take the
  // string up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = op->getAttrOfType<StringAttr>("string").getValue();
  // Ignore a leading ` in macro name.
  if (name.startswith("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(op->getResult(0), name);
}

void VerbatimExprOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getVerbatimExprAsmResultNames(getOperation(), std::move(setNameFn));
}

void VerbatimExprSEOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getVerbatimExprAsmResultNames(getOperation(), std::move(setNameFn));
}

//===----------------------------------------------------------------------===//
// ConstantXOp / ConstantZOp
//===----------------------------------------------------------------------===//

void ConstantXOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "x_i" << getWidth();
  setNameFn(getResult(), specialName.str());
}

LogicalResult ConstantXOp::verify() {
  // We don't allow zero width constant or unknown width.
  if (getWidth() <= 0)
    return emitError("unsupported type");
  return success();
}

void ConstantZOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "z_i" << getWidth();
  setNameFn(getResult(), specialName.str());
}

LogicalResult ConstantZOp::verify() {
  // We don't allow zero width constant or unknown type.
  if (getWidth() <= 0)
    return emitError("unsupported type");
  return success();
}

//===----------------------------------------------------------------------===//
// LocalParamOp
//===----------------------------------------------------------------------===//

void LocalParamOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the localparam has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

LogicalResult LocalParamOp::verify() {
  // Verify that this is a valid parameter value.
  return hw::checkParameterInContext(
      value(), (*this)->getParentOfType<hw::HWModuleOp>(), *this);
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

void RegOp::build(OpBuilder &builder, OperationState &odsState,
                  Type elementType, StringAttr name, StringAttr sym_name) {
  if (!name)
    name = builder.getStringAttr("");
  odsState.addAttribute("name", name);
  if (sym_name)
    odsState.addAttribute(hw::InnerName::getInnerNameAttrName(), sym_name);
  odsState.addTypes(hw::InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void RegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

// If this reg is only written to, delete the reg and all writers.
LogicalResult RegOp::canonicalize(RegOp op, PatternRewriter &rewriter) {
  // If the reg has a symbol, then we can't delete it.
  if (op.inner_symAttr())
    return failure();
  // Check that all operations on the wire are sv.assigns. All other wire
  // operations will have been handled by other canonicalization.
  for (auto &use : op.getResult().getUses())
    if (!isa<AssignOp>(use.getOwner()))
      return failure();

  // Remove all uses of the wire.
  for (auto &use : op.getResult().getUses())
    rewriter.eraseOp(use.getOwner());

  // Remove the wire.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IfDefOp

void IfDefOp::build(OpBuilder &builder, OperationState &result, StringRef cond,
                    std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  build(builder, result, builder.getStringAttr(cond), thenCtor, elseCtor);
}

void IfDefOp::build(OpBuilder &builder, OperationState &result, StringAttr cond,
                    std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  build(builder, result, MacroIdentAttr::get(builder.getContext(), cond),
        thenCtor, elseCtor);
}

void IfDefOp::build(OpBuilder &builder, OperationState &result,
                    MacroIdentAttr cond, std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute("cond", cond);
  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    builder.createBlock(elseRegion);
    elseCtor();
  }
}

static bool isEmptyBlockExceptForTerminator(Block *block) {
  assert(block && "Blcok must be non-null");
  return block->empty() || block->front().hasTrait<OpTrait::IsTerminator>();
}

// If both thenRegion and elseRegion are empty, erase op.
LogicalResult IfDefOp::canonicalize(IfDefOp op, PatternRewriter &rewriter) {
  if (!isEmptyBlockExceptForTerminator(op.getThenBlock()))
    return failure();

  if (op.hasElse() && !isEmptyBlockExceptForTerminator(op.getElseBlock()))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// IfDefProceduralOp

void IfDefProceduralOp::build(OpBuilder &builder, OperationState &result,
                              StringRef cond, std::function<void()> thenCtor,
                              std::function<void()> elseCtor) {
  IfDefOp::build(builder, result, cond, std::move(thenCtor),
                 std::move(elseCtor));
}

//===----------------------------------------------------------------------===//
// IfOp

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 std::function<void()> thenCtor,
                 std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(cond);
  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    builder.createBlock(elseRegion);
    elseCtor();
  }
}

/// Replaces the given op with the contents of the given single-block region.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *fromBlock = &region.front();
  // Merge it in above the specified operation.
  op->getBlock()->getOperations().splice(Block::iterator(op),
                                         fromBlock->getOperations());
}

LogicalResult IfOp::canonicalize(IfOp op, PatternRewriter &rewriter) {
  if (auto constant = op.cond().getDefiningOp<hw::ConstantOp>()) {

    if (constant.getValue().isAllOnesValue())
      replaceOpWithRegion(rewriter, op, op.thenRegion());
    else if (!op.elseRegion().empty())
      replaceOpWithRegion(rewriter, op, op.elseRegion());

    rewriter.eraseOp(op);

    return success();
  }

  // Erase empty if's.

  // If there is stuff in the then block, leave this operation alone.
  if (!op.getThenBlock()->empty())
    return failure();

  // If not and there is no else, then this operation is just useless.
  if (!op.hasElse() || op.getElseBlock()->empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  // Otherwise, invert the condition and move the 'else' block to the 'then'
  // region.
  auto cond = comb::createOrFoldNot(op.getLoc(), op.cond(), rewriter);
  op.setOperand(cond);

  auto *thenBlock = op.getThenBlock(), *elseBlock = op.getElseBlock();

  // Move the body of the then block over to the else.
  thenBlock->getOperations().splice(thenBlock->end(),
                                    elseBlock->getOperations());
  rewriter.eraseBlock(elseBlock);
  return success();
}

template <class Op>
static LogicalResult canonicalizeAlwaysLikeOp(Op op,
                                              PatternRewriter &rewriter) {
  // Erase empty op.
  if (!op.getBody()->empty())
    return failure();
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// AlwaysOp

AlwaysOp::Condition AlwaysOp::getCondition(size_t idx) {
  return Condition{EventControl(events()[idx].cast<IntegerAttr>().getInt()),
                   getOperand(idx)};
}

void AlwaysOp::build(OpBuilder &builder, OperationState &result,
                     ArrayRef<EventControl> events, ArrayRef<Value> clocks,
                     std::function<void()> bodyCtor) {
  assert(events.size() == clocks.size() &&
         "mismatch between event and clock list");
  OpBuilder::InsertionGuard guard(builder);

  SmallVector<Attribute> eventAttrs;
  for (auto event : events)
    eventAttrs.push_back(
        builder.getI32IntegerAttr(static_cast<int32_t>(event)));
  result.addAttribute("events", builder.getArrayAttr(eventAttrs));
  result.addOperands(clocks);

  // Set up the body.  Moves the insert point
  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (bodyCtor)
    bodyCtor();
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult AlwaysOp::verify() {
  if (events().size() != getNumOperands())
    return emitError("different number of operands and events");
  return success();
}

void AlwaysOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeAlwaysLikeOp<AlwaysOp>);
}

static ParseResult parseEventList(
    OpAsmParser &p, Attribute &eventsAttr,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &clocksOperands) {

  // Parse zero or more conditions intoevents and clocksOperands.
  SmallVector<Attribute> events;

  auto loc = p.getCurrentLocation();
  StringRef keyword;
  if (!p.parseOptionalKeyword(&keyword)) {
    while (1) {
      auto kind = symbolizeEventControl(keyword);
      if (!kind.hasValue())
        return p.emitError(loc, "expected 'posedge', 'negedge', or 'edge'");
      auto eventEnum = static_cast<int32_t>(kind.getValue());
      events.push_back(p.getBuilder().getI32IntegerAttr(eventEnum));

      clocksOperands.push_back({});
      if (p.parseOperand(clocksOperands.back()))
        return failure();

      if (failed(p.parseOptionalComma()))
        break;
      if (p.parseKeyword(&keyword))
        return failure();
    }
  }
  eventsAttr = p.getBuilder().getArrayAttr(events);
  return success();
}

static void printEventList(OpAsmPrinter &p, AlwaysOp op, ArrayAttr portsAttr,
                           OperandRange operands) {
  for (size_t i = 0, e = op.getNumConditions(); i != e; ++i) {
    if (i != 0)
      p << ", ";
    auto cond = op.getCondition(i);
    p << stringifyEventControl(cond.event);
    p << ' ';
    p.printOperand(cond.value);
  }
}

//===----------------------------------------------------------------------===//
// AlwaysFFOp

void AlwaysFFOp::build(OpBuilder &builder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute(
      "clockEdge", builder.getI32IntegerAttr(static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute(
      "resetStyle",
      builder.getI32IntegerAttr(static_cast<int32_t>(ResetType::NoReset)));

  // Set up the body.  Moves Insert Point
  builder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();

  // Set up the reset region.
  result.addRegion();
}

void AlwaysFFOp::build(OpBuilder &builder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       ResetType resetStyle, EventControl resetEdge,
                       Value reset, std::function<void()> bodyCtor,
                       std::function<void()> resetCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute(
      "clockEdge", builder.getI32IntegerAttr(static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute("resetStyle", builder.getI32IntegerAttr(
                                        static_cast<int32_t>(resetStyle)));
  result.addAttribute(
      "resetEdge", builder.getI32IntegerAttr(static_cast<int32_t>(resetEdge)));
  result.addOperands(reset);

  // Set up the body.  Moves Insert Point.
  builder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();

  // Set up the reset.  Moves Insert Point.
  builder.createBlock(result.addRegion());

  if (resetCtor)
    resetCtor();
}

LogicalResult AlwaysFFOp::canonicalize(AlwaysFFOp op,
                                       PatternRewriter &rewriter) {
  if (!op.getBody()->empty())
    return failure();
  if (!op.resetBlk().empty() && !op.getResetBlock()->empty())
  return failure();
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// AlwaysCombOp
//===----------------------------------------------------------------------===//

void AlwaysCombOp::build(OpBuilder &builder, OperationState &result,
                         std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  builder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();
}

void AlwaysCombOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add(canonicalizeAlwaysLikeOp<AlwaysCombOp>);
}

//===----------------------------------------------------------------------===//
// InitialOp
//===----------------------------------------------------------------------===//

void InitialOp::build(OpBuilder &builder, OperationState &result,
                      std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (bodyCtor)
    bodyCtor();
}

//===----------------------------------------------------------------------===//
// CaseOp
//===----------------------------------------------------------------------===//

/// Return the letter for the specified pattern bit, e.g. "0", "1", "x" or "z".
char sv::getLetter(CasePatternBit bit) {
  switch (bit) {
  case CasePatternBit::Zero:
    return '0';
  case CasePatternBit::One:
    return '1';
  case CasePatternBit::AnyX:
    return 'x';
  case CasePatternBit::AnyZ:
    return 'z';
  }
  llvm_unreachable("invalid casez PatternBit");
}

/// Return the specified bit, bit 0 is the least significant bit.
auto CasePattern::getBit(size_t bitNumber) const -> CasePatternBit {
  return CasePatternBit(unsigned(attr.getValue()[bitNumber * 2]) +
                        2 * unsigned(attr.getValue()[bitNumber * 2 + 1]));
}

bool CasePattern::isDefault() const {
  return attr.getValue().getBitWidth() % 2;
}

bool CasePattern::hasX() const {
  for (size_t i = 0, e = getWidth(); i != e; ++i)
    if (getBit(i) == CasePatternBit::AnyX)
      return true;
  return false;
}

bool CasePattern::hasZ() const {
  for (size_t i = 0, e = getWidth(); i != e; ++i)
    if (getBit(i) == CasePatternBit::AnyZ)
      return true;
  return false;
}

static SmallVector<CasePatternBit> getPatternBitsForValue(const APInt &value) {
  SmallVector<CasePatternBit> result;
  result.reserve(value.getBitWidth());
  for (size_t i = 0, e = value.getBitWidth(); i != e; ++i)
    result.push_back(CasePatternBit(value[i]));

  return result;
}

// Get a CasePattern from a specified list of PatternBits.  Bits are
// specified in most least significant order - element zero is the least
// significant bit.
CasePattern::CasePattern(const APInt &value, MLIRContext *context)
    : CasePattern(getPatternBitsForValue(value), context) {}

CasePattern::CasePattern(size_t width, DefaultPatternTag,
                         MLIRContext *context) {
  APInt pattern(width * 2 + 1, 0);
  auto patternType = IntegerType::get(context, width * 2 + 1);
  attr = IntegerAttr::get(patternType, pattern);
}

// Get a CasePattern from a specified list of PatternBits.  Bits are
// specified in most least significant order - element zero is the least
// significant bit.
CasePattern::CasePattern(ArrayRef<CasePatternBit> bits, MLIRContext *context) {
  APInt pattern(bits.size() * 2, 0);
  for (auto elt : llvm::reverse(bits)) {
    pattern <<= 2;
    pattern |= unsigned(elt);
  }
  auto patternType = IntegerType::get(context, bits.size() * 2);
  attr = IntegerAttr::get(patternType, pattern);
}

auto CaseOp::getCases() -> SmallVector<CaseInfo, 4> {
  SmallVector<CaseInfo, 4> result;
  assert(casePatterns().size() == getNumRegions() &&
         "case pattern / region count mismatch");
  size_t nextRegion = 0;
  for (auto elt : casePatterns()) {
    result.push_back({CasePattern(elt.cast<IntegerAttr>()),
                      &getRegion(nextRegion++).front()});
  }

  return result;
}

/// Parse case op.
/// case op ::= `sv.case` case-style? cond `:` type attr-dict  case-pattern^*
/// case-style ::= `case` | `casex` | `casez`
/// validation-qualifier (see SV Spec 12.5.3) ::= `unique` | `unique0`
///                                             | `priority`
/// case-pattern ::= `case` bit-pattern `:` region
ParseResult CaseOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::UnresolvedOperand condOperand;
  Type condType;

  auto loc = parser.getCurrentLocation();

  StringRef keyword;
  if (!parser.parseOptionalKeyword(&keyword)) {
    auto kind = symbolizeCaseStmtType(keyword);
    if (!kind.hasValue())
      return parser.emitError(loc, "expected 'case', 'casex', or 'casez'");
    auto caseEnum = static_cast<int32_t>(kind.getValue());
    result.addAttribute("caseStyle", builder.getI32IntegerAttr(caseEnum));
  }

  if (parser.parseOperand(condOperand) || parser.parseColonType(condType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(condOperand, condType, result.operands))
    return failure();

  // Check the integer type.
  if (!result.operands[0].getType().isSignlessInteger())
    return parser.emitError(loc, "condition must have signless integer type");
  auto condWidth = condType.getIntOrFloatBitWidth();

  // Parse all the cases.
  SmallVector<Attribute> casePatterns;
  SmallVector<CasePatternBit, 16> caseBits;
  bool defaultElem = false;
  while (1) {
    if (succeeded(parser.parseOptionalKeyword("default"))) {
      // Fill the pattern with Any.
      defaultElem = true;
    } else if (failed(parser.parseOptionalKeyword("case"))) {
      // Not default or case, must be the end of the cases.
      break;
    } else {
      // Parse the pattern.  It always starts with b, so it is an MLIR keyword.
      StringRef caseVal;
      loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&caseVal))
        return failure();

      if (caseVal.front() != 'b')
        return parser.emitError(loc, "expected case value starting with 'b'");
      caseVal = caseVal.drop_front();

      // Parse and decode each bit, we reverse the list later for MSB->LSB.
      for (; !caseVal.empty(); caseVal = caseVal.drop_front()) {
        CasePatternBit bit;
        switch (caseVal.front()) {
        case '0':
          bit = CasePatternBit::Zero;
          break;
        case '1':
          bit = CasePatternBit::One;
          break;
        case 'x':
          bit = CasePatternBit::AnyX;
          break;
        case 'z':
          bit = CasePatternBit::AnyZ;
          break;
        default:
          return parser.emitError(loc, "unexpected case bit '")
                 << caseVal.front() << "'";
        }
        caseBits.push_back(bit);
      }

      if (caseVal.size() > condWidth)
        return parser.emitError(loc, "too many bits specified in pattern");
      std::reverse(caseBits.begin(), caseBits.end());

      // High zeros may be missing.
      if (caseBits.size() < condWidth)
        caseBits.append(condWidth - caseBits.size(), CasePatternBit::Zero);
    }

    auto resultPattern =
        defaultElem ? CasePattern(condWidth, CasePattern::DefaultPatternTag(),
                                  builder.getContext())
                    : CasePattern(caseBits, builder.getContext());
    casePatterns.push_back(resultPattern.attr);
    caseBits.clear();

    // Parse the case body.
    auto caseRegion = std::make_unique<Region>();
    if (parser.parseColon() || parser.parseRegion(*caseRegion))
      return failure();
    result.addRegion(std::move(caseRegion));
  }

  result.addAttribute("casePatterns", builder.getArrayAttr(casePatterns));
  return success();
}

void CaseOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (caseStyle() == CaseStmtType::CaseXStmt)
    p << "casex ";
  else if (caseStyle() == CaseStmtType::CaseZStmt)
    p << "casez ";
  p << cond() << " : " << cond().getType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"casePatterns", "caseStyle"});

  for (auto caseInfo : getCases()) {
    p.printNewline();
    auto pattern = caseInfo.pattern;
    if (pattern.isDefault()) {
      p << "default";
    } else {
      p << "case b";
      for (size_t i = 0, e = pattern.getWidth(); i != e; ++i)
        p << getLetter(pattern.getBit(e - i - 1));
    }

    p << ": ";
    p.printRegion(*caseInfo.block->getParent(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

LogicalResult CaseOp::verify() {
  // Ensure that the number of regions and number of case values match.
  if (casePatterns().size() != getNumRegions())
    return emitOpError("case pattern / region count mismatch");
  return success();
}

/// This ctor allows you to build a CaseZ with some number of cases, getting
/// a callback for each case.
void CaseOp::build(OpBuilder &builder, OperationState &result,
                   CaseStmtType caseStyle, Value cond, size_t numCases,
                   std::function<CasePattern(size_t)> caseCtor) {
  result.addOperands(cond);
  result.addAttribute("caseStyle",
                      CaseStmtTypeAttr::get(builder.getContext(), caseStyle));
  SmallVector<Attribute> casePatterns;

  OpBuilder::InsertionGuard guard(builder);

  // Fill in the cases with the callback.
  for (size_t i = 0, e = numCases; i != e; ++i) {
    builder.createBlock(result.addRegion());
    casePatterns.push_back(caseCtor(i).attr);
  }

  result.addAttribute("casePatterns", builder.getArrayAttr(casePatterns));
}

// Strength reduce case styles based on the bit patterns.
LogicalResult CaseOp::canonicalize(CaseOp op, PatternRewriter &rewriter) {
  if (op.caseStyle() == CaseStmtType::CaseStmt)
    return failure();

  auto caseInfo = op.getCases();
  bool noXZ = llvm::all_of(caseInfo, [](const CaseInfo &ci) {
    return !ci.pattern.hasX() && !ci.pattern.hasZ();
  });
  bool noX = llvm::all_of(
      caseInfo, [](const CaseInfo &ci) { return !ci.pattern.hasX(); });
  bool noZ = llvm::all_of(
      caseInfo, [](const CaseInfo &ci) { return !ci.pattern.hasZ(); });

  if (op.caseStyle() == CaseStmtType::CaseXStmt) {
    if (noXZ) {
      rewriter.updateRootInPlace(op, [&]() {
        op.caseStyleAttr(
            CaseStmtTypeAttr::get(op.getContext(), CaseStmtType::CaseStmt));
      });
      return success();
    }
    if (noX) {
      rewriter.updateRootInPlace(op, [&]() {
        op.caseStyleAttr(
            CaseStmtTypeAttr::get(op.getContext(), CaseStmtType::CaseZStmt));
      });
      return success();
    }
  }

  if (op.caseStyle() == CaseStmtType::CaseZStmt && noZ) {
    rewriter.updateRootInPlace(op, [&]() {
      op.caseStyleAttr(
          CaseStmtTypeAttr::get(op.getContext(), CaseStmtType::CaseStmt));
    });
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Assignment statements
//===----------------------------------------------------------------------===//

LogicalResult BPAssignOp::verify() {
  if (isa<sv::WireOp>(dest().getDefiningOp()))
    return emitOpError(
        "Verilog disallows procedural assignment to a net type (did you intend "
        "to use a variable type, e.g., sv.reg?)");
  return success();
}

LogicalResult PAssignOp::verify() {
  if (isa<sv::WireOp>(dest().getDefiningOp()))
    return emitOpError(
        "Verilog disallows procedural assignment to a net type (did you intend "
        "to use a variable type, e.g., sv.reg?)");
  return success();
}

//===----------------------------------------------------------------------===//
// TypeDecl operations
//===----------------------------------------------------------------------===//

void InterfaceOp::build(OpBuilder &builder, OperationState &result,
                        StringRef sym_name, std::function<void()> body) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute(::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(sym_name));
  builder.createBlock(result.addRegion());
  if (body)
    body();
}

ModportType InterfaceOp::getModportType(StringRef modportName) {
  assert(lookupSymbol<InterfaceModportOp>(modportName) &&
         "Modport symbol not found.");
  auto *ctxt = getContext();
  return ModportType::get(
      getContext(),
      SymbolRefAttr::get(ctxt, sym_name(),
                         {SymbolRefAttr::get(ctxt, modportName)}));
}

Type InterfaceOp::getSignalType(StringRef signalName) {
  InterfaceSignalOp signal = lookupSymbol<InterfaceSignalOp>(signalName);
  assert(signal && "Interface signal symbol not found.");
  return signal.type();
}

static ParseResult parseModportStructs(OpAsmParser &parser,
                                       ArrayAttr &portsAttr) {

  auto context = parser.getBuilder().getContext();

  SmallVector<Attribute, 8> ports;
  auto parseElement = [&]() -> ParseResult {
    auto direction = ModportDirectionAttr::parse(parser, {});
    if (!direction)
      return failure();

    FlatSymbolRefAttr signal;
    if (parser.parseAttribute(signal))
      return failure();

    ports.push_back(ModportStructAttr::get(
        direction.cast<ModportDirectionAttr>(), signal, context));
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseElement))
    return failure();

  portsAttr = ArrayAttr::get(context, ports);
  return success();
}

static void printModportStructs(OpAsmPrinter &p, Operation *,
                                ArrayAttr portsAttr) {
  p << "(";
  llvm::interleaveComma(portsAttr, p, [&](Attribute attr) {
    auto port = attr.cast<ModportStructAttr>();
    p << stringifyEnum(port.direction().getValue());
    p << ' ';
    p.printSymbolName(port.signal().getRootReference().getValue());
  });
  p << ')';
}

void InterfaceSignalOp::build(mlir::OpBuilder &builder,
                              ::mlir::OperationState &state, StringRef name,
                              mlir::Type type) {
  build(builder, state, name, mlir::TypeAttr::get(type));
}

void InterfaceModportOp::build(OpBuilder &builder, OperationState &state,
                               StringRef name, ArrayRef<StringRef> inputs,
                               ArrayRef<StringRef> outputs) {
  auto *ctxt = builder.getContext();
  SmallVector<Attribute, 8> directions;
  ModportDirectionAttr inputDir =
      ModportDirectionAttr::get(ctxt, ModportDirection::input);
  ModportDirectionAttr outputDir =
      ModportDirectionAttr::get(ctxt, ModportDirection::output);
  for (auto input : inputs)
    directions.push_back(ModportStructAttr::get(
        inputDir, SymbolRefAttr::get(ctxt, input), ctxt));
  for (auto output : outputs)
    directions.push_back(ModportStructAttr::get(
        outputDir, SymbolRefAttr::get(ctxt, output), ctxt));
  build(builder, state, name, ArrayAttr::get(ctxt, directions));
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult InterfaceInstanceOp::verify() {
  auto *symtable = SymbolTable::getNearestSymbolTable(*this);
  if (!symtable)
    return emitError("sv.interface.instance must exist within a region "
                     "which has a symbol table.");
  auto ifaceTy = getType();
  auto referencedOp =
      SymbolTable::lookupSymbolIn(symtable, ifaceTy.getInterface());
  if (!referencedOp)
    return emitError("Symbol not found: ") << ifaceTy.getInterface() << ".";
  if (!isa<InterfaceOp>(referencedOp))
    return emitError("Symbol ")
           << ifaceTy.getInterface() << " is not an InterfaceOp.";
  return success();
}

/// Ensure that the symbol being instantiated exists and is an
/// InterfaceModportOp.
LogicalResult GetModportOp::verify() {
  auto *symtable = SymbolTable::getNearestSymbolTable(*this);
  if (!symtable)
    return emitError("sv.interface.instance must exist within a region "
                     "which has a symbol table.");
  auto ifaceTy = getType();
  auto referencedOp =
      SymbolTable::lookupSymbolIn(symtable, ifaceTy.getModport());
  if (!referencedOp)
    return emitError("Symbol not found: ") << ifaceTy.getModport() << ".";
  if (!isa<InterfaceModportOp>(referencedOp))
    return emitError("Symbol ")
           << ifaceTy.getModport() << " is not an InterfaceModportOp.";
  return success();
}

void GetModportOp::build(OpBuilder &builder, OperationState &state, Value value,
                         StringRef field) {
  auto ifaceTy = value.getType().dyn_cast<InterfaceType>();
  assert(ifaceTy && "GetModportOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(builder.getContext(), field);
  auto modportSym =
      SymbolRefAttr::get(ifaceTy.getInterface().getRootReference(), fieldAttr);
  build(builder, state, ModportType::get(builder.getContext(), modportSym),
        value, fieldAttr);
}

/// Lookup the op for the modport declaration.  This returns null on invalid
/// IR.
InterfaceModportOp
GetModportOp::getReferencedDecl(const hw::SymbolCache &cache) {
  return dyn_cast_or_null<InterfaceModportOp>(cache.getDefinition(fieldAttr()));
}

void ReadInterfaceSignalOp::build(OpBuilder &builder, OperationState &state,
                                  Value iface, StringRef signalName) {
  auto ifaceTy = iface.getType().dyn_cast<InterfaceType>();
  assert(ifaceTy && "ReadInterfaceSignalOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(builder.getContext(), signalName);
  InterfaceOp ifaceDefOp = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      iface.getDefiningOp(), ifaceTy.getInterface());
  assert(ifaceDefOp &&
         "ReadInterfaceSignalOp could not resolve an InterfaceOp.");
  build(builder, state, ifaceDefOp.getSignalType(signalName), iface, fieldAttr);
}

/// Lookup the op for the signal declaration.  This returns null on invalid
/// IR.
InterfaceSignalOp
ReadInterfaceSignalOp::getReferencedDecl(const hw::SymbolCache &cache) {
  return dyn_cast_or_null<InterfaceSignalOp>(
      cache.getDefinition(signalNameAttr()));
}

ParseResult parseIfaceTypeAndSignal(OpAsmParser &p, Type &ifaceTy,
                                    FlatSymbolRefAttr &signalName) {
  SymbolRefAttr fullSym;
  if (p.parseAttribute(fullSym) || fullSym.getNestedReferences().size() != 1)
    return failure();

  auto *ctxt = p.getBuilder().getContext();
  ifaceTy = InterfaceType::get(
      ctxt, FlatSymbolRefAttr::get(fullSym.getRootReference()));
  signalName = FlatSymbolRefAttr::get(fullSym.getLeafReference());
  return success();
}

void printIfaceTypeAndSignal(OpAsmPrinter &p, Operation *op, Type type,
                             FlatSymbolRefAttr signalName) {
  InterfaceType ifaceTy = type.dyn_cast<InterfaceType>();
  assert(ifaceTy && "Expected an InterfaceType");
  auto sym = SymbolRefAttr::get(ifaceTy.getInterface().getRootReference(),
                                {signalName});
  p << sym;
}

LogicalResult verifySignalExists(Value ifaceVal, FlatSymbolRefAttr signalName) {
  auto ifaceTy = ifaceVal.getType().dyn_cast<InterfaceType>();
  if (!ifaceTy)
    return failure();
  InterfaceOp iface = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      ifaceVal.getDefiningOp(), ifaceTy.getInterface());
  if (!iface)
    return failure();
  InterfaceSignalOp signal = iface.lookupSymbol<InterfaceSignalOp>(signalName);
  if (!signal)
    return failure();
  return success();
}

Operation *
InterfaceInstanceOp::getReferencedInterface(const hw::SymbolCache *cache) {
  FlatSymbolRefAttr interface = getInterfaceType().getInterface();
  if (cache)
    if (auto *result = cache->getDefinition(interface))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  return topLevelModuleOp.lookupSymbol(interface);
}

LogicalResult AssignInterfaceSignalOp::verify() {
  return verifySignalExists(iface(), signalNameAttr());
}

LogicalResult ReadInterfaceSignalOp::verify() {
  return verifySignalExists(iface(), signalNameAttr());
}

//===----------------------------------------------------------------------===//
// WireOp
//===----------------------------------------------------------------------===//

void WireOp::build(OpBuilder &builder, OperationState &odsState,
                   Type elementType, StringAttr name, StringAttr sym_name) {
  if (!name)
    name = builder.getStringAttr("");
  if (sym_name)
    odsState.addAttribute(hw::InnerName::getInnerNameAttrName(), sym_name);

  odsState.addAttribute("name", name);
  odsState.addTypes(InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void WireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

// If this wire is only written to, delete the wire and all writers.
LogicalResult WireOp::canonicalize(WireOp wire, PatternRewriter &rewriter) {
  // If the wire has a symbol, then we can't delete it.
  if (wire.inner_symAttr())
    return failure();

  // Wires have inout type, so they'll have assigns and read_inout operations
  // that work on them.  If anything unexpected is found then leave it alone.
  SmallVector<sv::ReadInOutOp> reads;
  sv::AssignOp write;

  for (auto *user : wire->getUsers()) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user)) {
      reads.push_back(read);
      continue;
    }

    // Otherwise must be an assign, and we must not have seen a write yet.
    auto assign = dyn_cast<sv::AssignOp>(user);
    // Either the wire has more than one write or another kind of Op (other than
    // AssignOp and ReadInOutOp), then can't optimize.
    if (!assign || write)
      return failure();
    write = assign;
  }

  Value connected;
  if (!write) {
    // If no write and only reads, then replace with XOp.
    connected = rewriter.create<ConstantXOp>(
        wire.getLoc(),
        wire.getResult().getType().cast<InOutType>().getElementType());
  } else if (isa<hw::HWModuleOp>(write->getParentOp()))
    connected = write.src();
  else
    // If the write is happening at the module level then we don't have any
    // use-before-def checking to do, so we only handle that for now.
    return failure();

  // Ok, we can do this.  Replace all the reads with the connected value.
  for (auto read : reads)
    rewriter.replaceOp(read, connected);

  // And remove the write and wire itself.
  if (write)
    rewriter.eraseOp(write);
  rewriter.eraseOp(wire);
  return success();
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult WireOp::verify() {
  if (!isa<hw::HWModuleOp>((*this)->getParentOp()))
    return emitError("sv.wire must not be in an always or initial block");
  return success();
}

//===----------------------------------------------------------------------===//
// ReadInOutOp
//===----------------------------------------------------------------------===//

void ReadInOutOp::build(OpBuilder &builder, OperationState &result,
                        Value input) {
  auto resultType = input.getType().cast<InOutType>().getElementType();
  build(builder, result, resultType, input);
}

//===----------------------------------------------------------------------===//
// ArrayIndexInOutOp
//===----------------------------------------------------------------------===//

void ArrayIndexInOutOp::build(OpBuilder &builder, OperationState &result,
                              Value input, Value index) {
  auto resultType = input.getType().cast<InOutType>().getElementType();
  resultType = getAnyHWArrayElementType(resultType);
  assert(resultType && "input should have 'inout of an array' type");
  build(builder, result, InOutType::get(resultType), input, index);
}

//===----------------------------------------------------------------------===//
// IndexedPartSelectInOutOp
//===----------------------------------------------------------------------===//

void IndexedPartSelectInOutOp::build(OpBuilder &builder, OperationState &result,
                                     Value input, Value base, int32_t width,
                                     bool decrement) {
  auto resultType =
      hw::InOutType::get(IntegerType::get(builder.getContext(), width));
  build(builder, result, resultType, input, base, width, decrement);
}

LogicalResult IndexedPartSelectInOutOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::RegionRange regions,
    SmallVectorImpl<Type> &results) {
  auto width = attrs.get("width");
  if (!width)
    return failure();

  results.push_back(hw::InOutType::get(
      IntegerType::get(context, width.cast<IntegerAttr>().getInt())));
  return success();
}

LogicalResult IndexedPartSelectInOutOp::verify() {
  unsigned inputWidth = 0, resultWidth = 0;
  auto opWidth = width();

  if (auto i = input()
                   .getType()
                   .cast<InOutType>()
                   .getElementType()
                   .dyn_cast<IntegerType>())
    inputWidth = i.getWidth();
  else
    return emitError("input element type must be Integer");

  if (auto resType =
          getType().cast<InOutType>().getElementType().dyn_cast<IntegerType>())
    resultWidth = resType.getWidth();
  else
    return emitError("result element type must be Integer");

  if (opWidth > inputWidth)
    return emitError("slice width should not be greater than input width");
  if (opWidth != resultWidth)
    return emitError("result width must be equal to slice width");
  return success();
}

OpFoldResult IndexedPartSelectInOutOp::fold(ArrayRef<Attribute> constants) {
  if (getType() == input().getType())
    return input();
  return {};
}

//===----------------------------------------------------------------------===//
// IndexedPartSelectOp
//===----------------------------------------------------------------------===//

void IndexedPartSelectOp::build(OpBuilder &builder, OperationState &result,
                                Value input, Value base, int32_t width,
                                bool decrement) {
  auto resultType = (IntegerType::get(builder.getContext(), width));
  build(builder, result, resultType, input, base, width, decrement);
}

LogicalResult IndexedPartSelectOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::RegionRange regions,
    SmallVectorImpl<Type> &results) {
  auto width = attrs.get("width");
  if (!width)
    return failure();

  results.push_back(
      IntegerType::get(context, width.cast<IntegerAttr>().getInt()));
  return success();
}

LogicalResult IndexedPartSelectOp::verify() {
  auto opWidth = width();

  unsigned resultWidth = getType().cast<IntegerType>().getWidth();
  unsigned inputWidth = input().getType().cast<IntegerType>().getWidth();

  if (opWidth > inputWidth)
    return emitError("slice width should not be greater than input width");
  if (opWidth != resultWidth)
    return emitError("result width must be equal to slice width");
  return success();
}

//===----------------------------------------------------------------------===//
// StructFieldInOutOp
//===----------------------------------------------------------------------===//

LogicalResult StructFieldInOutOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::RegionRange regions,
    SmallVectorImpl<Type> &results) {
  auto field = attrs.get("field");
  if (!field)
    return failure();
  auto structType =
      hw::type_cast<hw::StructType>(getInOutElementType(operands[0].getType()));
  auto resultType = structType.getFieldType(field.cast<StringAttr>());
  if (!resultType)
    return failure();

  results.push_back(hw::InOutType::get(resultType));
  return success();
}

//===----------------------------------------------------------------------===//
// Other ops.
//===----------------------------------------------------------------------===//

LogicalResult AliasOp::verify() {
  // Must have at least two operands.
  if (operands().size() < 2)
    return emitOpError("alias must have at least two operands");

  return success();
}

//===----------------------------------------------------------------------===//
// PAssignOp
//===----------------------------------------------------------------------===//

// reg s <= cond ? val : s simplification.
// Don't assign a register's value to itself, conditionally assign the new value
// instead.
LogicalResult PAssignOp::canonicalize(PAssignOp op, PatternRewriter &rewriter) {
  auto mux = op.src().getDefiningOp<comb::MuxOp>();
  if (!mux)
    return failure();

  auto reg = dyn_cast<sv::RegOp>(op.dest().getDefiningOp());
  if (!reg)
    return failure();

  bool trueBranch; // did we find the register on the true branch?
  auto tvread = mux.trueValue().getDefiningOp<sv::ReadInOutOp>();
  auto fvread = mux.falseValue().getDefiningOp<sv::ReadInOutOp>();
  if (tvread && reg == tvread.input().getDefiningOp<sv::RegOp>())
    trueBranch = true;
  else if (fvread && reg == fvread.input().getDefiningOp<sv::RegOp>())
    trueBranch = false;
  else
    return failure();

  // Check that this is the only write of the register
  for (auto &use : reg->getUses()) {
    if (isa<ReadInOutOp>(use.getOwner()))
      continue;
    if (use.getOwner() == op)
      continue;
    return failure();
  }

  // Replace a non-blocking procedural assign in a procedural region with a
  // conditional procedural assign.  We've ensured that this is the only write
  // of the register.
  if (trueBranch) {
    auto cond = comb::createOrFoldNot(mux.getLoc(), mux.cond(), rewriter);
    rewriter.create<sv::IfOp>(mux.getLoc(), cond, [&]() {
      rewriter.create<PAssignOp>(op.getLoc(), reg, mux.falseValue());
    });
  } else {
    rewriter.create<sv::IfOp>(mux.getLoc(), mux.cond(), [&]() {
      rewriter.create<PAssignOp>(op.getLoc(), reg, mux.trueValue());
    });
  }

  // Remove the wire.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// BindOp
//===----------------------------------------------------------------------===//

/// Instances must be at the top level of the hw.module (or within a `ifdef)
// and are typically at the end of it, so we scan backwards to find them.
template <class Op>
static Op findInstanceSymbolInBlock(StringAttr name, Block *body) {
  for (auto &op : llvm::reverse(body->getOperations())) {
    if (auto instance = dyn_cast<Op>(op)) {
      if (instance.inner_sym() &&
          instance.inner_sym().getValue() == name.getValue())
        return instance;
    }

    if (auto ifdef = dyn_cast<IfDefOp>(op)) {
      if (auto result =
              findInstanceSymbolInBlock<Op>(name, ifdef.getThenBlock()))
        return result;
      if (ifdef.hasElse())
        if (auto result =
                findInstanceSymbolInBlock<Op>(name, ifdef.getElseBlock()))
          return result;
    }
  }
  return {};
}

hw::InstanceOp BindOp::getReferencedInstance(const hw::SymbolCache *cache) {
  // If we have a cache, directly look up the referenced instance.
  if (cache) {
    auto result = cache->getDefinition(instance());
    return cast<hw::InstanceOp>(result.getOp());
  }

  // Otherwise, resolve the instance by looking up the module ...
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return {};

  auto hwModule = dyn_cast_or_null<hw::HWModuleOp>(
      topLevelModuleOp.lookupSymbol(instance().getModule()));
  if (!hwModule)
    return {};

  // ... then look up the instance within it.
  return findInstanceSymbolInBlock<hw::InstanceOp>(instance().getName(),
                                                   hwModule.getBodyBlock());
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult BindOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<mlir::ModuleOp>();
  auto hwModule = dyn_cast_or_null<hw::HWModuleOp>(
      symbolTable.lookupSymbolIn(module, instance().getModule()));
  if (!hwModule)
    return emitError("Referenced module doesn't exist ")
           << instance().getModule() << "::" << instance().getName();

  auto inst = findInstanceSymbolInBlock<hw::InstanceOp>(
      instance().getName(), hwModule.getBodyBlock());
  if (!inst)
    return emitError("Referenced instance doesn't exist ")
           << instance().getModule() << "::" << instance().getName();
  if (!inst->getAttr("doNotPrint"))
    return emitError("Referenced instance isn't marked as doNotPrint");
  return success();
}

void BindOp::build(OpBuilder &builder, OperationState &odsState, StringAttr mod,
                   StringAttr name) {
  auto ref = hw::InnerRefAttr::get(mod, name);
  odsState.addAttribute("instance", ref);
}

//===----------------------------------------------------------------------===//
// BindInterfaceOp
//===----------------------------------------------------------------------===//

sv::InterfaceInstanceOp
BindInterfaceOp::getReferencedInstance(const hw::SymbolCache *cache) {
  // If we have a cache, directly look up the referenced instance.
  if (cache) {
    auto result = cache->getDefinition(instance());
    return cast<sv::InterfaceInstanceOp>(result.getOp());
  }

  // Otherwise, resolve the instance by looking up the module ...
  auto *symbolTable = SymbolTable::getNearestSymbolTable(*this);
  if (!symbolTable)
    return {};
  auto *parentOp =
      lookupSymbolInNested(symbolTable, instance().getModule().getValue());
  if (!parentOp)
    return {};

  // ... then look up the instance within it.
  return findInstanceSymbolInBlock<sv::InterfaceInstanceOp>(
      instance().getName(), &parentOp->getRegion(0).front());
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult
BindInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto parentOp =
      symbolTable.lookupNearestSymbolFrom(*this, instance().getModule());
  if (!parentOp)
    return emitError("Referenced module doesn't exist ")
           << instance().getModule() << "::" << instance().getName();

  auto inst = findInstanceSymbolInBlock<sv::InterfaceInstanceOp>(
      instance().getName(), &parentOp->getRegion(0).front());
  if (!inst)
    return emitError("Referenced interface doesn't exist ")
           << instance().getModule() << "::" << instance().getName();
  if (!inst->getAttr("doNotPrint"))
    return emitError("Referenced interface isn't marked as doNotPrint");
  return success();
}

//===----------------------------------------------------------------------===//
// XMROp
//===----------------------------------------------------------------------===//

ParseResult parseXMRPath(::mlir::OpAsmParser &parser, ArrayAttr &pathAttr,
                         StringAttr &terminalAttr) {
  SmallVector<Attribute> strings;
  ParseResult ret = parser.parseCommaSeparatedList([&]() {
    StringAttr result;
    StringRef keyword;
    if (succeeded(parser.parseOptionalKeyword(&keyword))) {
      strings.push_back(parser.getBuilder().getStringAttr(keyword));
      return success();
    }
    if (succeeded(parser.parseAttribute(
            result, parser.getBuilder().getType<NoneType>()))) {
      strings.push_back(result);
      return success();
    }
    return failure();
  });
  if (succeeded(ret)) {
    pathAttr = parser.getBuilder().getArrayAttr(
        ArrayRef(strings.begin(), strings.end() - 1));
    terminalAttr = (*strings.rbegin()).cast<StringAttr>();
  }
  return ret;
}

void printXMRPath(OpAsmPrinter &p, XMROp op, ArrayAttr pathAttr,
                  StringAttr terminalAttr) {
  llvm::interleaveComma(pathAttr, p);
  p << ", " << terminalAttr;
}

//===----------------------------------------------------------------------===//
// Verification Ops.
//===----------------------------------------------------------------------===//

static LogicalResult eraseIfNotZero(Operation *op, Value value,
                                    PatternRewriter &rewriter) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    if (!constant.getValue().isZero()) {
      rewriter.eraseOp(op);
      return success();
    }

  return failure();
}

template <class Op>
static LogicalResult canonicalizeImmediateVerifOp(Op op,
                                                  PatternRewriter &rewriter) {
  return eraseIfNotZero(op, op.expression(), rewriter);
}

void AssertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<AssertOp>);
}

void AssumeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<AssumeOp>);
}

void CoverOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<CoverOp>);
}

template <class Op>
static LogicalResult canonicalizeConcurrentVerifOp(Op op,
                                                   PatternRewriter &rewriter) {
  return eraseIfNotZero(op, op.property(), rewriter);
}

void AssertConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add(canonicalizeConcurrentVerifOp<AssertConcurrentOp>);
}

void AssumeConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add(canonicalizeConcurrentVerifOp<AssumeConcurrentOp>);
}

void CoverConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add(canonicalizeConcurrentVerifOp<CoverConcurrentOp>);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
