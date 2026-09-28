# test_simple_ragas.py
import os
from datasets import Dataset
from ragas import evaluate
from config.settings import settings

# Set environment variable for RAGAS
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Your data
data = {
    "question": ["What is deep learning?"],
    "answer": ["Deep learning is a subset of machine learning."],
    "contexts": [["Deep learning uses neural networks."]],
    "ground_truth": ["Deep learning is machine learning with neural networks."]
}

dataset = Dataset.from_dict(data)

# Simple evaluate - RAGAS auto-configures everything
result = evaluate(dataset)

print("Result:", result)