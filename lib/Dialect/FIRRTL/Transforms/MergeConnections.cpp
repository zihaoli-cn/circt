//===- MergeConnections.cpp - Merge expanded connections --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass converts registers that are reset to an invalid value to resetless
// registers.  This is a reduced implementation of the Scala FIRRTL Compiler's
// MergeConnections pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "firrtl-merge-connections"

using namespace circt;
using namespace firrtl;

namespace {
// Return ture if it might be fine to merge the connect.
static bool okToPeel(ConnectOp connect) {
  // Ignore connections between different types.
  if (connect.src().getType() != connect.dest().getType())
    return false;
  // Ignore if either source or destination is an argument.
  if (!connect.dest().getDefiningOp() || !connect.src().getDefiningOp())
    return false;
  return true;
}

// Return the constant value if value is essentially constant.
static Value getConstant(Value value) {
  if (auto constant = value.getDefiningOp<ConstantOp>())
    return constant;
  if (auto constant = value.getDefiningOp<InvalidValueOp>())
    return constant;
  if (auto bitcast = value.getDefiningOp<BitCastOp>())
    return getConstant(bitcast.input());

  // TODO: Add unrealized_conversion, asUInt, asSInt
  return {};
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

// A helper struct to merge connections.
struct MergeConnection {
  bool peelFullyConnect(ConnectOp connect);

  bool peelConstantConnect(ConnectOp connect);

  template <class AggregateType>
  void tryFullyConnect(AggregateType type, Value src, Value dest);

  template <class AggregateType>
  void tryConstantConnect(AggregateType type, Value dest);

  void run(FModuleOp moduleOp);

  // Stats.
  unsigned numMergedConnections = 0;
  unsigned numNewConnections = 0;

  // A set to track investigated destinations.
  DenseSet<FieldRef> visitedDestination;

  // A map from a destination FieldRef to connect op.
  DenseMap<FieldRef, ConnectOp> fieldRefToConnect;

  // A map from a destination FieldRef to fieldRef of their source value.
  DenseMap<FieldRef, FieldRef> connectDestMap;

  // A map from a destination FieldRef to Constant-like value for their source
  // value.
  DenseMap<FieldRef, Value> connectToConstantMap;

  // A work list for fully connection merging. We use a queue to traverse leafs
  // first.
  std::queue<std::pair<Value, Value>> fullyConnectionWorklist;

  // A work list for constant connection merging. We use a queue to traverse
  // leafs first.
  std::queue<Value> constantConnectWorklist;

  // Container for unused connections to erase in the post-process.
  SmallVector<ConnectOp> shouldBeRemoved;
};

// Peel if the connect might be a part of full connections.
bool MergeConnection::peelFullyConnect(ConnectOp connect) {
  if (!okToPeel(connect))
    return false;
  auto *dest = connect.dest().getDefiningOp();
  auto *src = connect.src().getDefiningOp();
  if (!((isa<SubindexOp>(dest) && isa<SubindexOp>(src)) ||
        (isa<SubfieldOp>(dest) && isa<SubfieldOp>(src))))
    return false;
  auto destFieldRef = getFieldRefFromValue(connect.dest());
  auto srcFieldRef = getFieldRefFromValue(connect.dest());

  connectDestMap[destFieldRef] = srcFieldRef;
  fieldRefToConnect[destFieldRef] = connect;

  fullyConnectionWorklist.push({src->getOperand(0), dest->getOperand(0)});
  return true;
}

// Push the connection to worklists if the connect might be a part of full
// connections.
bool MergeConnection::peelConstantConnect(ConnectOp connect) {
  if (!okToPeel(connect))
    return false;
  auto *dest = connect.dest().getDefiningOp();
  if (!(isa<SubindexOp>(dest) || (isa<SubfieldOp>(dest))))
    return false;
  auto constant = getConstant(connect.src());
  if (!constant)
    return false;

  auto destFieldRef = getFieldRefFromValue(connect.dest());
  connectToConstantMap[destFieldRef] = constant;
  fieldRefToConnect[destFieldRef] = connect;
  constantConnectWorklist.push(dest->getOperand(0));
  return true;
}

// Connect src and dest directly if src and dest are fully connected.
template <class AggregateType>
void MergeConnection::tryFullyConnect(AggregateType type, Value src,
                                      Value dest) {
  auto srcFieldRef = getFieldRefFromValue(src);
  auto destFieldRef = getFieldRefFromValue(dest);

  // Skip if we have already looked at the dest.
  if (!visitedDestination.insert(destFieldRef).second)
    return;

  // Check all sub elements are connected.
  for (auto idx : llvm::seq(0u, (unsigned)type.getNumElements())) {
    auto offset = type.getFieldID(idx);
    auto it = connectDestMap.find(destFieldRef.getSubField(offset));
    if (it == connectDestMap.end() ||
        it->second != srcFieldRef.getSubField(offset)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Fail to merge because they are not fully connected\n");
      return;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Merge Success\n";);

  // Ok, all subelements of `src` and `dest` are connected properly. Therefore,
  // it is fine to connect `src` to `dest`. So remove all existing conections
  // between .
  for (auto idx : llvm::seq(0u, (unsigned)type.getNumElements())) {
    auto offset = type.getFieldID(idx);
    auto connect = fieldRefToConnect[destFieldRef.getSubField(offset)];
    LLVM_DEBUG(llvm::dbgs() << "Erase:" << connect << "\n";);
    shouldBeRemoved.push_back(connect);
  }

  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(dest.getLoc(), src.getParentBlock());

  auto newConnection = builder.create<ConnectOp>(dest, src);
  LLVM_DEBUG(llvm::dbgs() << "New connection:" << newConnection << "\n";);

  // New connection might be merged as well.
  peelFullyConnect(newConnection);
  ++numNewConnections;
}

template <class AggregateType>
void MergeConnection::tryConstantConnect(AggregateType type, Value dest) {
  auto destFieldRef = getFieldRefFromValue(dest);

  // Skip if we have already looked at the dest.
  if (!visitedDestination.insert(destFieldRef).second)
    return;

  // Check all sub elements are connected to constants.
  for (auto idx : llvm::seq(0u, (unsigned)type.getNumElements())) {
    auto offset = type.getFieldID(idx);
    if (!connectToConstantMap.count(destFieldRef.getSubField(offset))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Fail to merge because they are not fully connected\n");
      return;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Merge Success\n";);

  // Ok, all subelements are connected to constant so let's connect dest to
  // concatnation {c1, c2, c3, c4, ..., cn}. For simplicity, we first bitcast
  // each constant to uint with same bitwidth, and then bitcast the concatnation
  // into the type of destination.
  Value accumulate;
  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(dest.getLoc(), dest.getParentBlock());
  for (auto idx : llvm::seq(0u, (unsigned)type.getNumElements())) {
    auto offset = type.getFieldID(idx);
    auto constant = connectToConstantMap[destFieldRef.getSubField(offset)];

    constant = builder.createOrFold<BitCastOp>(
        UIntType::get(constant.getContext(),
                      *firrtl::getBitWidth(
                          constant.getType().template cast<FIRRTLType>())),
        constant);

    accumulate =
        (accumulate ? builder.createOrFold<CatPrimOp>(accumulate, constant)
                    : constant);

    auto connnect = fieldRefToConnect[destFieldRef.getSubField(offset)];
    LLVM_DEBUG(llvm::dbgs() << "Erase:" << connnect << "\n";);
    shouldBeRemoved.push_back(connnect);
    numMergedConnections++;
  }

  accumulate = builder.createOrFold<BitCastOp>(dest.getType(), accumulate);
  auto newConnection = builder.create<ConnectOp>(dest, accumulate);
  LLVM_DEBUG(llvm::dbgs() << "New connection:" << newConnection << "\n";);
  // New connection might be merged as well.
  peelConstantConnect(newConnection);
  ++numNewConnections;
}

void MergeConnection::run(FModuleOp moduleOp) {
  // Peel all connects.
  moduleOp.walk([&](ConnectOp connect) {
    LLVM_DEBUG(llvm::dbgs() << "Trying to peel: " << connect << "\n");
    if (peelFullyConnect(connect) || peelConstantConnect(connect))
      return;
  });

  // Run fully connection merging.
  while (!fullyConnectionWorklist.empty()) {
    auto [src, dest] = fullyConnectionWorklist.front();
    fullyConnectionWorklist.pop();

    LLVM_DEBUG(llvm::dbgs() << "=== Trying fully connect merging:\nSRC: " << src
                            << "\nDEST: " << dest << "\n");
    // We cannot merge if the type is not same.
    if (src.getType() != dest.getType()) {
      LLVM_DEBUG(llvm::dbgs() << "Fail to merge because of type mismatch\n");
      continue;
    }

    if (auto bundle = src.getType().dyn_cast<BundleType>())
      tryFullyConnect(bundle, src, dest);
    else if (auto vector = src.getType().dyn_cast<FVectorType>())
      tryFullyConnect(vector, src, dest);
  }

  // Run constant connection merging.
  while (!constantConnectWorklist.empty()) {
    auto dest = constantConnectWorklist.front();
    constantConnectWorklist.pop();
    LLVM_DEBUG(llvm::dbgs() << "=== Trying constant connect merging:"
                            << "\nDEST: " << dest << "\n");

    if (auto bundle = dest.getType().dyn_cast<BundleType>())
      tryConstantConnect(bundle, dest);
    else if (auto vector = dest.getType().dyn_cast<FVectorType>())
      tryConstantConnect(vector, dest);
  }

  // Remove dead connections.
  for (auto connect : shouldBeRemoved)
    connect.erase();

  // Clean up subfield/subindex.
  auto body = moduleOp.getBody();
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body)))
    if (isa<SubfieldOp, SubindexOp, InvalidValueOp, ConstantOp, BitCastOp>(op))
      if (op.use_empty())
        op.erase();
}

struct MergeConnectionsPass
    : public MergeConnectionsBase<MergeConnectionsPass> {
  void runOnOperation() override;
};
} // namespace

void MergeConnectionsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Running MergeConnections "
                             "--------------------------------------===\n"
                          << "Module: '" << getOperation().getName() << "'\n";);

  MergeConnection mergeConnection;
  mergeConnection.run(getOperation());
  numMergedConnections += mergeConnection.numMergedConnections;
  numNewConnections += mergeConnection.numNewConnections;

  if (mergeConnection.numNewConnections == 0)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createMergeConnectionsPass() {
  return std::make_unique<MergeConnectionsPass>();
}
