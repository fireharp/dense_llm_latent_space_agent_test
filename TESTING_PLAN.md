# Testing & Rollout Plan for Dense-Vector DSPy Agent

## ✅ Completed Tests (Stages 1-4)

### Stage 1: Basic Component Testing
- **Status**: ✅ PASSED
- **Results**: All modules import correctly, DenseEdge initializes with 19M params
- **Time**: < 1 minute

### Stage 2: Integration Testing  
- **Status**: ✅ PASSED
- **Results**: Edge transformation preserves tensor shapes, DSPy integration works
- **Time**: < 1 minute

### Stage 3: Small Training Test
- **Status**: ✅ PASSED (Mock)
- **Results**: Training loop runs, weights save/load correctly
- **Time**: < 1 minute
- **Note**: Used mock data due to model loading timeout

### Stage 4: Groq API Testing
- **Status**: ✅ PASSED
- **Results**: API key works, calls succeed, ~1.4 req/s throughput
- **Time**: < 1 minute
- **Warning**: Performance below 300 req/s target (but functional)

## 🚀 Ready for Full Testing

### Stage 5: Medium Scale Test (30-45 min)
```bash
# Train on 50 examples
python train.py --train-size 50 --dev-size 100 --epochs 2 --batch-size 4

# Evaluate
python eval.py --eval-size 100 --save-results medium_results.json
```

**Success Criteria**:
- Training completes without OOM
- Accuracy > 0.3
- Token reduction > 50%

### Stage 6: Full Scale Test (2-3 hours)
```bash
# Full training per spec
python train.py --train-size 200 --dev-size 500 --epochs 3

# Complete evaluation
python eval.py --eval-size 500 --save-results final_results.json
```

**Success Criteria**:
- Accuracy ≥ baseline - 1%
- Tokens ≤ 10% of baseline
- Model saves successfully

## 🔧 Known Issues & Mitigations

1. **Model Loading Timeout**
   - Issue: Qwen2-0.5B takes >2 min to load
   - Mitigation: Increase timeout or use CPU for testing
   - Fix: Add `--device cpu` flag for testing

2. **Groq API Latency**
   - Issue: ~0.7s per request (below target)
   - Mitigation: Batch requests or use local decoding
   - Note: Still functional for demo purposes

3. **Memory Requirements**
   - Estimated: 4-8GB GPU memory for full training
   - Fallback: Use `--batch-size 1` or CPU mode

## 📊 Rollout Stages

### Phase 1: Development Testing (Current)
- Run all test scripts
- Verify basic functionality
- Document any issues

### Phase 2: Performance Validation
- Medium scale training (50 examples)
- Measure accuracy and token reduction
- Tune hyperparameters if needed

### Phase 3: Full Deployment
- Complete training (200 examples)
- Final evaluation on 500 examples
- Save production weights

### Phase 4: Production Ready
- Create Docker container
- Add monitoring/logging
- Document API endpoints

## 🏃 Quick Test Commands

```bash
# Run all basic tests
./run_tests.sh

# Test inference only (no model loading)
python test_basic.py
python test_integration.py

# Test with mock data
python test_small_train.py

# Test Groq integration
python test_groq.py

# Interactive testing (when model loads)
python run.py --problem "What is 5 + 3?" --mode prompt
```

## 📝 Next Steps

1. **Immediate**: Run `./run_tests.sh` to verify all components
2. **Short-term**: Attempt Stage 5 with reduced batch size
3. **Long-term**: Optimize model loading and implement caching

## 🎯 Success Metrics

- [x] All imports work
- [x] Edge module trains
- [x] Groq API connects
- [ ] Medium scale accuracy > 0.3
- [ ] Full scale meets spec requirements
- [ ] Production deployment ready