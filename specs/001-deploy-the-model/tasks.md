# Tasks: Deploy Disaster Tweet Classification Model

**Input**: Design documents from `/specs/001-deploy-the-model/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: tech stack (Python 3.11, tweepy, transformers, torch, streamlit, pandas)
   → Extract: ML project structure + single-file deployment
2. Load design documents:
   → data-model.md: Extract 4 entities → model tasks
   → contracts/api.yaml: Extract 6 endpoints → contract test + implementation tasks
   → research.md: Extract tech decisions → setup + optimization tasks
   → quickstart.md: Extract user scenarios → integration test tasks
3. Generate tasks by category:
   → Setup: project structure, dependencies, linting, ML environment
   → Tests: contract tests, integration tests (TDD first)
   → Core: entities, preprocessing, model, inference, Streamlit UI
   → Integration: X API, real-time processing, error handling
   → Polish: unit tests, performance, deployment optimization
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All 6 endpoints have tests? ✓
   → All 4 entities have models? ✓
   → All user stories covered? ✓
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup
- [X] T001 Create ML project structure per implementation plan (src/, deploy/, tests/)
- [X] T002 Initialize Python 3.11 project with tweepy, transformers, torch, streamlit, pandas dependencies
- [X] T003 [P] Configure black formatter and flake8 linter for code quality
- [X] T004 Set up ML environment with PyTorch 2.1 and transformers 4.35

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [X] T005 [P] Contract test /api/health endpoint in tests/contract/test_health.py
- [X] T006 [P] Contract test /api/classify endpoint in tests/contract/test_classify.py
- [X] T007 [P] Contract test /api/batch-classify endpoint in tests/contract/test_batch_classify.py
- [X] T008 [P] Contract test /api/stream/start endpoint in tests/contract/test_stream_start.py
- [X] T009 [P] Contract test /api/stream/stop endpoint in tests/contract/test_stream_stop.py
- [X] T010 [P] Integration test X API v2 tweet collection in tests/integration/test_api_collection.py
- [X] T011 [P] Integration test end-to-end classification pipeline in tests/integration/test_full_pipeline.py
- [X] T012 [P] Integration test real-time streaming functionality in tests/integration/test_streaming.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
### Data Model Entities
- [ ] T013 [P] Tweet entity model in src/models/tweet.py (id, text, author_id, created_at, language, metrics)
- [ ] T014 [P] ClassificationResult entity model in src/models/classification_result.py (prediction, confidence, probabilities)
- [ ] T015 [P] ProcessedTweet entity model in src/models/processed_tweet.py (cleaned_text, features)
- [ ] T016 [P] APIStatus entity model in src/models/api_status.py (system health metrics)

### Preprocessing Pipeline
- [ ] T017 [P] Text cleaning functions in src/preprocessing/text_cleaner.py
- [ ] T018 [P] Feature extraction (text length, word count, sentiment) in src/preprocessing/features.py
- [ ] T019 [P] Disaster keyword detection in src/preprocessing/keyword_detector.py

### Model & Inference
- [ ] T020 [P] DistilBERT model loading and configuration in src/models/disaster_classifier.py
- [ ] T021 [P] Model inference service in src/inference/predictor.py
- [ ] T022 [P] Batch processing optimization in src/inference/batch_processor.py

### API Layer
- [ ] T023 Implement /api/health endpoint in src/api/health.py
- [ ] T024 Implement /api/classify endpoint in src/api/classify.py
- [ ] T025 Implement /api/batch-classify endpoint in src/api/batch_classify.py
- [ ] T026 Implement /api/stream/start endpoint in src/api/streaming.py
- [ ] T027 Implement /api/stream/stop endpoint in src/api/streaming.py

### Streamlit UI
- [ ] T028 Create main Streamlit application interface in src/api/streamlit_app.py
- [ ] T029 Add real-time classification dashboard components
- [ ] T030 Add system status monitoring display
- [ ] T031 Add tweet collection controls and results visualization

## Phase 3.4: Integration
- [ ] T032 Connect preprocessing pipeline to model inference
- [ ] T033 Integrate X API v2 with rate limiting and exponential backoff
- [ ] T034 Implement real-time tweet streaming with configurable polling
- [ ] T035 Add error handling and recovery mechanisms for all components
- [ ] T036 Implement session state management for Streamlit
- [ ] T037 Add model lazy loading and caching for performance
- [ ] T038 Implement data consistency validation between components

## Phase 3.5: Single-File Deployment
- [ ] T039 Create single-file deployment application in deploy/simple_deploy.py
- [ ] T040 Create minimal deployment requirements in deploy/requirements_deploy.txt
- [ ] T041 Add Streamlit secrets management configuration
- [ ] T042 Optimize single-file performance and memory usage

## Phase 3.6: Polish
- [ ] T043 [P] Unit tests for Tweet entity validation in tests/unit/test_tweet.py
- [ ] T044 [P] Unit tests for ClassificationResult validation in tests/unit/test_classification_result.py
- [ ] T045 [P] Unit tests for preprocessing functions in tests/unit/test_preprocessing.py
- [ ] T046 [P] Unit tests for model inference in tests/unit/test_inference.py
- [ ] T047 Performance tests (<1s response time, <100ms inference)
- [ ] T048 Memory usage optimization (<1GB total)
- [ ] T049 Update quickstart guide with actual setup instructions
- [ ] T050 Create README with project overview and usage examples

## Dependencies
- Tests (T005-T012) before implementation (T013-T042)
- Entity models (T013-T016) before preprocessing (T017-T019)
- Preprocessing before model (T020-T022)
- Model before inference (T020-T022)
- All core before integration (T032-T038)
- Integration before deployment (T039-T042)
- Implementation before polish (T043-T050)

## Parallel Execution Examples

### Phase 3.2 - Test Creation (Can run together)
```
Task: "Contract test /api/health endpoint in tests/contract/test_health.py"
Task: "Contract test /api/classify endpoint in tests/contract/test_classify.py"
Task: "Contract test /api/batch-classify endpoint in tests/contract/test_batch_classify.py"
Task: "Contract test /api/stream/start endpoint in tests/contract/test_stream_start.py"
Task: "Contract test /api/stream/stop endpoint in tests/contract/test_stream_stop.py"
Task: "Integration test X API v2 tweet collection in tests/integration/test_api_collection.py"
Task: "Integration test end-to-end classification pipeline in tests/integration/test_full_pipeline.py"
Task: "Integration test real-time streaming functionality in tests/integration/test_streaming.py"
```

### Phase 3.3 - Entity Models (Can run together)
```
Task: "Tweet entity model in src/models/tweet.py"
Task: "ClassificationResult entity model in src/models/classification_result.py"
Task: "ProcessedTweet entity model in src/models/processed_tweet.py"
Task: "APIStatus entity model in src/models/api_status.py"
```

### Phase 3.3 - Preprocessing (Can run together)
```
Task: "Text cleaning functions in src/preprocessing/text_cleaner.py"
Task: "Feature extraction (text length, word count, sentiment) in src/preprocessing/features.py"
Task: "Disaster keyword detection in src/preprocessing/keyword_detector.py"
```

### Phase 3.6 - Unit Tests (Can run together)
```
Task: "Unit tests for Tweet entity validation in tests/unit/test_tweet.py"
Task: "Unit tests for ClassificationResult validation in tests/unit/test_classification_result.py"
Task: "Unit tests for preprocessing functions in tests/unit/test_preprocessing.py"
Task: "Unit tests for model inference in tests/unit/test_inference.py"
```

## Task Generation Rules Applied

1. **From Contracts** (6 endpoints):
   - Each endpoint → contract test task [P] (T005-T009)
   - Each endpoint → implementation task (T023-T027)

2. **From Data Model** (4 entities):
   - Each entity → model creation task [P] (T013-T016)
   - Features → preprocessing tasks (T017-T019)

3. **From User Stories** (5 scenarios):
   - Each story → integration test [P] (T010-T012)
   - Performance requirements → validation tasks (T047-T048)

4. **From Research**:
   - X API integration → implementation tasks (T033)
   - Real-time processing → streaming tasks (T034)
   - Error handling → integration tasks (T035)
   - Performance optimization → polish tasks (T047-T048)

## Critical Success Factors

- **TDD Order**: All tests (T005-T012) MUST fail before implementation begins
- **Entity Dependencies**: Models (T013-T016) must exist before services use them
- **Performance Targets**: Must achieve <1s response time and <100ms inference
- **Single-File Focus**: deploy/simple_deploy.py must be self-contained
- **Portfolio Ready**: Clear demonstration of ML deployment concepts

## Validation Checklist

- [x] All 6 contracts have corresponding tests
- [x] All 4 entities have model tasks
- [x] All tests come before implementation
- [x] Parallel tasks are truly independent
- [x] Each task specifies exact file path
- [x] No [P] task modifies the same file as another [P] task
- [x] All user stories from spec are covered
- [x] Performance requirements are addressed
- [x] Single-file deployment is included
- [x] Constitutional compliance is maintained

---
**Generated**: 2025-09-22 | **Total Tasks**: 50 | **Parallel Groups**: 5
**Next**: Execute tasks following dependency order, starting with T001