//===- HWCleanup.cpp - HW Cleanup Pass ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs various cleanups and canonicalization
// transformations for hw.module bodies.  This is intended to be used early in
// the HW/SV pipeline to expose optimization opportunities that require global
// analysis.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace mlir;
using namespace matchers;
using namespace comb;

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

namespace {

/// Check the equivalence of operations by doing a deep comparison of operands
/// and attributes, but does not compare the content of any regions attached to
/// each op.
struct AlwaysLikeOpInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return mlir::OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/mlir::OperationEquivalence::directHashValue,
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    // Trivially the same.
    if (lhs == rhs)
      return true;
    // Filter out tombstones and empty ops.
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    // Compare attributes.
    if (lhs->getName() != rhs->getName() ||
        lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
        lhs->getNumOperands() != rhs->getNumOperands())
      return false;
    // Compare operands.
    for (auto operandPair : llvm::zip(lhs->getOperands(), rhs->getOperands())) {
      Value lhsOperand = std::get<0>(operandPair);
      Value rhsOperand = std::get<1>(operandPair);
      if (lhsOperand != rhsOperand)
        return false;
    }
    // The two AlwaysOps are similar enough to be combined.
    return true;
  }
};

} // end anonymous namespace

// Merge two regions together. These regions must only have a one block.
static void mergeRegions(Region *region1, Region *region2) {
  assert(region1->getBlocks().size() <= 1 && region2->getBlocks().size() <= 1 &&
         "Can only merge regions with a single block");
  if (region1->empty()) {
    // If both regions are empty, move on to the next pair of regions
    if (region2->empty())
      return;
    // If the first region has no block, move the second region's block over.
    region1->getBlocks().splice(region1->end(), region2->getBlocks());
    return;
  }

  // If the second region is not empty, splice its block into the start of the
  // first region.
  if (!region2->empty()) {
    auto &block1 = region1->front();
    auto &block2 = region2->front();
    block1.getOperations().splice(block1.begin(), block2.getOperations());
  }
}

// This struct analyses given two conditions.
// cond1 = conj1 /\ commonConj
// cond2 = conj2 /\ commonConj
struct ConditionPairInformation {
  ConditionPairInformation(Value cond1, Value cond2) {
    peelConjunction(cond1, conj1);
    peelConjunction(cond2, conj2);

    for (auto c1 : conj1)
      if (conj2.contains(c1))
        commonConj.push_back(c1);

    if (commonConj.empty())
      return;

    // If conj1 is equal to conj2, all conditions are common.
    if (commonConj.size() == conj1.size() &&
        commonConj.size() == conj2.size()) {
      conj1.clear();
      conj2.clear();
      return;
    }

    // Remove all common values from both sets.
    for (auto c : commonConj) {
      conj1.remove(c);
      conj2.remove(c);
    }
  }

  void cleanUpUses() {
    for (Operation *op : andOperations)
      if (op->use_empty())
        op->erase();
  }

  // Set to store unique values in the first condition.
  llvm::SmallSetVector<Value, 4> conj1;
  // Set to store unique values in the second condition.
  llvm::SmallSetVector<Value, 4> conj2;
  // Store common values of two conditions.
  llvm::SmallVector<Value> commonConj;

private:
  // Track decomposed operations which might be dead after if-merging.
  llvm::SmallSetVector<comb::AndOp, 4> andOperations;

  // Helper function to accmulate conditions.
  void peelConjunction(Value cond,
                       llvm::SmallSetVector<Value, 4> &conjunctions) {
    auto andOp = dyn_cast_or_null<comb::AndOp>(cond.getDefiningOp());
    if (!andOp) {
      conjunctions.insert(cond);
      return;
    }

    andOperations.insert(andOp);
    llvm::for_each(andOp.getOperands(),
                   [&](Value cond) { peelConjunction(cond, conjunctions); });
  }
};

//===----------------------------------------------------------------------===//

// HWCleanupPass
//===----------------------------------------------------------------------===//

namespace {
struct HWCleanupPass : public sv::HWCleanupBase<HWCleanupPass> {
  void runOnOperation() override;

  void runOnRegionsInOp(Operation &op);
  void runOnGraphRegion(Region &region);
  void runOnProceduralRegion(Region &region);

private:
  /// Inline all regions from the second operation into the first and delete the
  /// second operation.
  void mergeOperationsIntoFrom(Operation *op1, Operation *op2) {
    assert(op1 != op2 && "Cannot merge an op into itself");
    for (size_t i = 0, e = op1->getNumRegions(); i != e; ++i)
      mergeRegions(&op1->getRegion(i), &op2->getRegion(i));

    op2->erase();
    anythingChanged = true;
  }

  sv::IfOp mergeIfOpConditions(sv::IfOp op1, sv::IfOp op2);

  sv::IfOp mergeIfOps(sv::IfOp ifOp, sv::IfOp prevIfOp) {
    assert(ifOp != prevIfOp && "Cannot merge an op into itself");

    // If conditions of op1 and op2 are equal, we can just merge them.
    if (ifOp.cond() == prevIfOp.cond()) {
      anythingChanged = true;
      mergeOperationsIntoFrom(ifOp, prevIfOp);
      return ifOp;
    }

    // TOOD: Handle even when they have else blocks
    if (ifOp.hasElse() || prevIfOp.hasElse())
      return ifOp;
    if(getenv("disable"))
    return ifOp;

    return mergeIfOpConditions(ifOp, prevIfOp);
  }

  void runIfOpToCasezOnBlock(Block &block);

  bool anythingChanged;
};
} // end anonymous namespace

void HWCleanupPass::runOnOperation() {
  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;
  runOnGraphRegion(getOperation().getBody());

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

/// Recursively process all of the regions in the specified op, dispatching to
/// graph or procedural processing as appropriate.
void HWCleanupPass::runOnRegionsInOp(Operation &op) {
  if (op.hasTrait<sv::ProceduralRegion>()) {
    for (auto &region : op.getRegions())
      runOnProceduralRegion(region);
  } else {
    for (auto &region : op.getRegions())
      runOnGraphRegion(region);
  }
}

/// Run simplifications on the specified graph region.
void HWCleanupPass::runOnGraphRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  // A set of operations in the current block which are mergable. Any
  // operation in this set is a candidate for another similar operation to
  // merge in to.
  DenseSet<Operation *, AlwaysLikeOpInfo> alwaysFFOpsSeen;
  llvm::SmallDenseMap<Attribute, Operation *, 4> ifdefOps;
  sv::InitialOp initialOpSeen;
  sv::AlwaysCombOp alwaysCombOpSeen;

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Merge alwaysff and always operations by hashing them to check to see if
    // we've already encountered one.  If so, merge them and reprocess the body.
    if (isa<sv::AlwaysOp, sv::AlwaysFFOp>(op)) {
      // Merge identical alwaysff's together and delete the old operation.
      auto itAndInserted = alwaysFFOpsSeen.insert(&op);
      if (itAndInserted.second)
        continue;
      auto *existingAlways = *itAndInserted.first;
      mergeOperationsIntoFrom(&op, existingAlways);

      *itAndInserted.first = &op;
      continue;
    }

    // Merge graph ifdefs anywhere in the module.
    if (auto ifdefOp = dyn_cast<sv::IfDefOp>(op)) {
      auto *&entry = ifdefOps[ifdefOp.condAttr()];
      if (entry)
        mergeOperationsIntoFrom(ifdefOp, entry);

      entry = ifdefOp;
      continue;
    }

    // Merge initial ops anywhere in the module.
    if (auto initialOp = dyn_cast<sv::InitialOp>(op)) {
      if (initialOpSeen)
        mergeOperationsIntoFrom(initialOp, initialOpSeen);
      initialOpSeen = initialOp;
      continue;
    }

    // Merge always_comb ops anywhere in the module.
    if (auto alwaysComb = dyn_cast<sv::AlwaysCombOp>(op)) {
      if (alwaysCombOpSeen)
        mergeOperationsIntoFrom(alwaysComb, alwaysCombOpSeen);
      alwaysCombOpSeen = alwaysComb;
      continue;
    }
  }

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op.
    if (op.getNumRegions() != 0)
      runOnRegionsInOp(op);
  }
}

/// Run simplifications on the specified procedural region.
void HWCleanupPass::runOnProceduralRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  Operation *lastSideEffectingOp = nullptr;
  for (auto it = body.begin(), end = body.end(); it != end; it++) {
    Operation &op = *it;
    // Merge procedural ifdefs with neighbors in the procedural region.
    if (auto ifdef = dyn_cast<sv::IfDefProceduralOp>(op)) {
      if (auto prevIfDef =
              dyn_cast_or_null<sv::IfDefProceduralOp>(lastSideEffectingOp)) {
        if (ifdef.cond() == prevIfDef.cond()) {
          // We know that there are no side effective operations between the
          // two, so merge the first one into this one.
          mergeOperationsIntoFrom(ifdef, prevIfDef);
        }
      }
    }

    // Merge 'if' operations.
    if (auto ifop = dyn_cast<sv::IfOp>(op)) {
      if (auto prevIf = dyn_cast_or_null<sv::IfOp>(lastSideEffectingOp)) {
        auto newIfOp = mergeIfOps(ifop, prevIf);
        // If new operation is returned, we have to update the iterator.
        if (newIfOp != ifop) {
          lastSideEffectingOp = newIfOp;
          it = newIfOp->getIterator();
          continue;
        }
      }
    }

    // Keep track of the last side effecting operation we've seen.
    if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op))
      lastSideEffectingOp = &op;
  }

  runIfOpToCasezOnBlock(body);

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op.
    if (op.getNumRegions() != 0)
      runOnRegionsInOp(op);
  }
}

// This function merge two if operations when they have a common condition.
// For example,
// prevIfOp:
//   if a1 & a2 &... & an & c1 & c2 .. & cn {e1}
// ifOp:
//   if b1 & b2 &... & bn & c1 & c2 .. & cn {e2}
// ====>
//   if c1 & c2 .. & cn {
//     if a1 & a2 & ... & an {e1}
//     if b1 & b2 & ... & bn {e2}
//   }
sv::IfOp HWCleanupPass::mergeIfOpConditions(sv::IfOp ifOp, sv::IfOp prevIfOp) {

  ConditionPairInformation condPairInfo(ifOp.cond(), prevIfOp.cond());

  // If there is nothing in common, we cannot merge.
  if (condPairInfo.commonConj.empty())
    return ifOp;

  // If both conjunction1 and conjunction2 are empty, it means the conditions
  // are actually equivalent.
  if (condPairInfo.conj1.empty() && condPairInfo.conj2.empty()) {
    mergeOperationsIntoFrom(ifOp, prevIfOp);
    return ifOp;
  }

  OpBuilder builder(ifOp.getContext());
  auto i1Type = ifOp.cond().getType();

  auto generateCondValue =
      [&](Location loc, llvm::SmallSetVector<Value, 4> &conjunction) -> Value {
    if (conjunction.empty())
      return builder.create<hw::ConstantOp>(loc, i1Type, 1);

    return builder.createOrFold<comb::AndOp>(loc, i1Type,
                                             conjunction.takeVector());
  };

  // If conj2 is empty, it means we can move ifOp to the block of prevIfOp.
  if (condPairInfo.conj2.empty()) {
    // Move ifOp to the end of prevIfOp's then block.
    builder.setInsertionPointToEnd(prevIfOp.getThenBlock());
    auto newCond1 = generateCondValue(ifOp.cond().getLoc(), condPairInfo.conj1);
    ifOp.setOperand(newCond1);
    // op1 might contain ops defined between op2 and op1, we have to move op2
    // to the position of op1 to ensure that the dominance doesn't break.
    prevIfOp->moveBefore(ifOp);
    ifOp->moveBefore(prevIfOp.getThenBlock(), prevIfOp.getThenBlock()->end());
    condPairInfo.cleanUpUses();
    return prevIfOp;
  }

  // If conj1 is empty, it means we can move prevIfOp to the block of ifOp.
  if (condPairInfo.conj1.empty()) {
    // Move prevIfOp to the start of ifOp's then block.
    builder.setInsertionPointToStart(ifOp.getThenBlock());
    auto newCond2 =
        generateCondValue(prevIfOp.cond().getLoc(), condPairInfo.conj2);
    prevIfOp.setOperand(newCond2);
    prevIfOp->moveAfter(ifOp.getThenBlock(), ifOp.getThenBlock()->begin());
    condPairInfo.cleanUpUses();
    return ifOp;
  }

  // Ok, now we create a new if op to contain ifOp and prevIfOp.
  builder.setInsertionPoint(ifOp);
  auto newCond1 = generateCondValue(ifOp.getLoc(), condPairInfo.conj1);
  auto newCond2 = generateCondValue(prevIfOp.getLoc(), condPairInfo.conj2);
  auto cond =
      builder.createOrFold<comb::AndOp>(ifOp.getLoc(), condPairInfo.commonConj);
  auto newIf = builder.create<sv::IfOp>(prevIfOp.getLoc(), cond, [&]() {});

  prevIfOp->moveBefore(newIf.getThenBlock(), newIf.getThenBlock()->begin());
  ifOp->moveAfter(prevIfOp);
  ifOp.setOperand(newCond1);
  prevIfOp.setOperand(newCond2);
  condPairInfo.cleanUpUses();

  return newIf;
}

void HWCleanupPass::runIfOpToCasezOnBlock(Block &body) {
  Value comparedValue;
  SmallVector<std::pair<circt::hw::ConstantOp, sv::IfOp>> constantOp;
  llvm::MapVector<APInt, SmallVector<sv::IfOp>> newMap;

  auto constructCaseZ = [&](Operation *op) {
    if (newMap.size() <= 1 || !comparedValue) {
      newMap.clear();
      comparedValue = nullptr;
      return;
    }

    auto constantOp = newMap.takeVector();

    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);
    auto casez = builder.create<sv::CaseZOp>(
        comparedValue.getLoc(), comparedValue, constantOp.size(),
        [&](size_t caseIdx) -> circt::sv::CaseZPattern {
          auto constant = constantOp[caseIdx].first;

          circt::sv::CaseZPattern thePattern =
              circt::sv::CaseZPattern(constant, op->getContext());
          return thePattern;
        });

    auto cases = casez.getCases();
    unsigned idx = 0;
    for (unsigned idx : llvm::seq(0ul, constantOp.size())) {
      auto block = cases[idx].block;
      auto ifOps = constantOp[idx].second;
      for (auto ifOp : llvm::reverse(ifOps)) {
        block->getOperations().splice(std::prev(block->end()),
                                      ifOp.getThenBlock()->getOperations());
        ifOp.erase();
      }
      idx++;
    }

    constantOp.clear();
    newMap = llvm::MapVector<APInt, SmallVector<sv::IfOp>>();
    comparedValue = nullptr;
  };

  Operation *op = nullptr;
  for (auto it = body.begin(), end = body.end(); it != end; it++) {
    op = &*it;
    if (auto ifop = dyn_cast<sv::IfOp>(op)) {
      if (!ifop.hasElse()) {
        if (auto icmpOp =
                dyn_cast_or_null<comb::ICmpOp>(ifop.cond().getDefiningOp())) {
          if (icmpOp.predicate() == comb::ICmpPredicate::eq) {
            if (auto constant = dyn_cast_or_null<circt::hw::ConstantOp>(
                    icmpOp.rhs().getDefiningOp())) {
              if (!comparedValue || comparedValue == icmpOp.lhs()) {
                if (!comparedValue)
                  comparedValue = icmpOp.lhs();
                newMap[constant.getValue()].push_back(ifop);
                continue;
              }
            }
          }
        }
      }
    }
    if (!mlir::MemoryEffectOpInterface::hasNoEffect(op))
      constructCaseZ(op);
  }

  if (op)
    constructCaseZ(op);
}
std::unique_ptr<Pass> circt::sv::createHWCleanupPass() {
  return std::make_unique<HWCleanupPass>();
}
