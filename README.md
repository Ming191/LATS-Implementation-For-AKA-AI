# Python LATS Test Generation Server

Unified test generation server using Language Agent Tree Search (LATS) with Monte Carlo Tree Search for C++ MC/DC coverage.

## Architecture

```
Single API Endpoint → LATS Orchestrator → MCTS Loop
                                        ↓
                    ┌─────────────────────────────┐
                    │  Selection → Expansion      │
                    │  Simulation → Backprop      │
                    └─────────────────────────────┘
                                        ↓
                    Reflection Agent ← Java Execution
```

## Project Structure

```
python-lats-server/
├── core/
│   └── lats/
│       ├── state.py         # TestState, TestCase, ConditionInfo
│       ├── tree.py          # TreeNode, UCB1 logic
│       ├── mcts.py          # MCTS controller (TODO)
│       ├── orchestrator.py  # Main orchestrator (TODO)
│       ├── reflection.py    # Reflection agent (TODO)
│       └── reward.py        # Reward function (TODO)
├── api/
│   └── routes.py            # FastAPI endpoints (TODO)
├── models/
│   └── schemas.py           # Request/response models (TODO)
├── prompts/
│   └── unified/             # Unified prompt templates (TODO)
└── tests/
    └── test_*.py            # Unit tests (TODO)
```

## Implementation Status

### ✅ Phase 1a: Core Infrastructure (COMPLETED)
- [x] `TestState` - Stateless state representation (names only)
- [x] `TestCase` - Test case metadata model
- [x] `ConditionInfo` - Coverage condition tracking
- [x] `ExecutionResult` - Cumulative execution feedback
- [x] `TreeNode` - MCTS tree node with UCB1
- [x] `ExecutionEngine` - Java backend integration with caching
- [x] `RewardFunction` - Progressive reward computation
- [x] `ContextManager` - Session-based context caching
- [x] Unit tests (50+ test cases)
- [x] Project structure setup

### ✅ Phase 1b: MCTS Controller (COMPLETED)
- [x] `MCTSController` class with 4-phase MCTS
- [x] Selection phase (UCB1 traversal to leaf)
- [x] Expansion phase (K candidate generation with adaptive K)
- [x] Simulation phase (execution + reward computation)
- [x] Backpropagation phase (ancestor update)
- [x] Adaptive K (5 → 3 → 1 based on progress)
- [x] Beam search pruning (keep top-5 branches)
- [x] Early termination (coverage target, no progress, token budget)
- [x] Integration tests (15+ test cases)

### ✅ Phase 2: LLM Integration (COMPLETED - Week 1)
- [x] DeepSeek LLM client with retry logic (exponential backoff)
- [x] Prompt templates (initialize, expand_batch, expand_targeted, reflection)
- [x] Candidate generation in MCTS (initialize/batch/targeted modes)
- [x] Adaptive temperature (0.9 → 0.7 → 0.5)
- [x] JSON response parsing with markdown handling
- [x] Token tracking integration (100k budget)
- [x] Error learning (recent errors fed to LLM)
- [x] Test deduplication

### 🚧 Phase 3: API Layer & Reflection (CURRENT)
- [ ] FastAPI routes for /api/v1/lats/search
- [ ] Pydantic request/response models
- [ ] Reflection agent implementation (failure analysis)
- [ ] Audit logging system

### 📋 Upcoming Phases
- Phase 4: Java GUI integration
- Phase 5: Performance optimization
- Phase 6: Production deployment

## Design Decisions

1. **Unified Approach**: No foundation/expansion separation - single continuous search
2. **MC/DC Only**: Always target MC/DC coverage (superset of branch/statement)
3. **No Coverage Type Parameter**: Simplified API - only `coverage_target` (0.0-1.0)
4. **Progressive Generation**: MCTS naturally handles initial → expansion in one flow
5. **Stateless Python**: Test code managed by Java TestCaseManager - Python only tracks names
6. **Cumulative Coverage**: Java computes coverage via CoverageManager.getCoverageOfMultiTestCaseAtFunctionLevel()
7. **Incremental Execution**: New tests executed individually, cumulative coverage computed from .tp files

## Next Steps

Week 1 Complete! ✅ Core LATS with LLM integration implemented.

**Next:** Build FastAPI API layer and test with real C++ functions.
