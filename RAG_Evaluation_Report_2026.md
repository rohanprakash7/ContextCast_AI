# RAG System Evaluation Report

**Generated:** 2026-04-13 15:00:23

**Video:** kY14KfZQ1TI 

**Questions Evaluated:** 10

---

## Executive Summary

**Best Configuration:** `simple_k6`

- **Overall Score:** 0.908
- **Faithfulness:** 1.000 (Perfect = 1.0)
- **Context Precision:** 0.870
- **Context Recall:** 0.836

### Key Findings

- ✓ Hybrid retriever improved Context Recall to 0.836
- ✓ RECOMMENDED: Use simple_k6 for best overall performance

---

## Detailed Results

### All Configurations Tested

| Configuration | Faithfulness | Precision | Recall | Overall Score |
|--------------|--------------|-----------|--------|---------------|
| simple_k4    | 0.936 | 0.958 | 0.744 | 0.864 |
| simple_k6    | 1.000 | 0.870 | 0.836 | 0.908 |
| hybrid_k4    | 0.966 | 0.920 | 0.817 | 0.897 |
| hybrid_k6    | 0.993 | 0.770 | 0.836 | 0.886 |


### Improvements Over Baseline

**Baseline Configuration:** `simple_k4`

| Configuration | Faithfulness | Precision | Recall |
|--------------|--------------|-----------|--------|
| simple_k6    | +0.064 | -0.088 | +0.092 |
| hybrid_k4    | +0.030 | -0.038 | +0.073 |
| hybrid_k6    | +0.057 | -0.188 | +0.092 |

---

## Metric Explanations

### Faithfulness (0.0 - 1.0)
Measures if answers are grounded in retrieved context without hallucination.
- **1.0:** Perfect - No hallucinations
- **0.8-1.0:** Excellent
- **< 0.8:** Needs improvement

### Context Precision (0.0 - 1.0)
Measures relevance of retrieved chunks to the question.
- **> 0.9:** Excellent - Retrieving very relevant chunks
- **0.7-0.9:** Good
- **< 0.7:** Needs better retrieval

### Context Recall (0.0 - 1.0)
Measures completeness - did we retrieve all relevant information?
- **> 0.8:** Excellent - Very complete retrieval
- **0.6-0.8:** Good
- **< 0.6:** Missing important information

---

## Recommendations

### Recommended Configuration: `simple_k6`

**Why this configuration?**

- Achieves best balance of all metrics (Overall Score: 0.908)
- Perfect faithfulness (1.000) - No hallucinations
- Strong recall (0.836) - Retrieves most relevant information
- Good precision (0.870) - Retrieved chunks are relevant

### Implementation Steps

```python
# Update your RAG system to use the best configuration:
result = answer_question(
    question=question,
    video_id=video_id,
    retriever_type='simple',
    top_k=6
)
```

---

## Configuration Details

### 1. simple_k4

- **Retriever Type:** simple
- **Top K:** 4
- **Faithfulness:** 0.936
- **Context Precision:** 0.958
- **Context Recall:** 0.744
-  Good recall but could be improved


### 2. simple_k6

- **Retriever Type:** simple
- **Top K:** 6
- **Faithfulness:** 1.000
- **Context Precision:** 0.870
- **Context Recall:** 0.836
- Excellent faithfulness - minimal hallucination
- Excellent recall - retrieving complete information


### 3. hybrid_k4

- **Retriever Type:** hybrid
- **Top K:** 4
- **Faithfulness:** 0.966
- **Context Precision:** 0.920
- **Context Recall:** 0.817
-  Excellent faithfulness - minimal hallucination
-  Excellent recall - retrieving complete information


### 4. hybrid_k6

- **Retriever Type:** hybrid
- **Top K:** 6
- **Faithfulness:** 0.993
- **Context Precision:** 0.770
- **Context Recall:** 0.836
-  Excellent faithfulness - minimal hallucination
-  Excellent recall - retrieving complete information


---


*Report generated on 2026-04-13 15:00:23*