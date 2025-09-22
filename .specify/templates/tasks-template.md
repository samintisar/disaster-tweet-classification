# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **ML Project**: `src/preprocessing/`, `src/models/`, `src/inference/`, `tests/`
- **Single file deployment**: `deploy/simple_deploy.py`, `tests/`
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume ML project structure - adjust based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools
- [ ] T004 Set up ML environment (PyTorch, transformers, etc.)

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T005 [P] Contract test text preprocessing in tests/contract/test_preprocessing.py
- [ ] T006 [P] Contract test feature extraction in tests/contract/test_features.py
- [ ] T007 [P] Integration test model inference in tests/integration/test_inference.py
- [ ] T008 [P] Integration test end-to-end tweet classification in tests/integration/test_classification.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T009 [P] Text preprocessing functions in src/preprocessing/text_cleaner.py
- [ ] T010 [P] Feature extraction in src/preprocessing/features.py
- [ ] T011 [P] Model loading in src/models/disaster_classifier.py
- [ ] T012 [P] Inference service in src/inference/predictor.py
- [ ] T013 [P] Streamlit UI in src/api/streamlit_app.py
- [ ] T014 Integration testing and performance validation

## Phase 3.4: Integration
- [ ] T015 Connect preprocessing to model inference
- [ ] T016 X API v2 integration for real-time tweets
- [ ] T017 Performance monitoring and optimization
- [ ] T018 Error handling and logging for ML components

## Phase 3.5: Polish
- [ ] T019 [P] Unit tests for validation in tests/unit/test_validation.py
- [ ] T020 Performance tests (<200ms)
- [ ] T021 [P] Update docs/api.md
- [ ] T022 Remove duplication
- [ ] T023 Run manual-testing.md

## Dependencies
- Tests (T005-T008) before implementation (T009-T014)
- T009 blocks T010, T015
- T011 blocks T012
- Implementation before polish (T019-T023)

## Parallel Example
```
# Launch T005-T008 together:
Task: "Contract test text preprocessing in tests/contract/test_preprocessing.py"
Task: "Contract test feature extraction in tests/contract/test_features.py"
Task: "Integration test model inference in tests/integration/test_inference.py"
Task: "Integration test end-to-end tweet classification in tests/integration/test_classification.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P]
   - Each ML component → implementation task

2. **From Data Model**:
   - Each entity → model creation task [P]
   - Features → preprocessing tasks

3. **From User Stories**:
   - Each story → integration test [P]
   - ML performance requirements → validation tasks

4. **Ordering**:
   - Setup → Tests → Preprocessing → Models → Inference → UI → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All contracts have corresponding tests
- [ ] All entities have model tasks
- [ ] All tests come before implementation
- [ ] Parallel tasks truly independent
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task