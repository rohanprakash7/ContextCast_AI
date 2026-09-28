"""
Evaluation Test Dataset - MIT 6.S094 Deep Learning for Self-Driving Cars
High-quality test questions for RAG evaluation using RAGAS
Generated from MIT 6.S094 Deep Learning lecture transcript
Focus: Deep Learning fundamentals WITH self-driving car applications
"""

from typing import List, Dict

# Enhanced test questions for video kY14KfZQ1TI
# Balanced: Deep Learning concepts + Self-Driving Car applications
# Mixed difficulty: Basic, Intermediate, Advanced

TEST_QUESTIONS = [
    # BASIC - Course Overview & Fundamentals (3 questions)
    "What is the 6.S094 course about and what will students learn?",
    "What is deep learning and how does it automate pattern extraction from data?",
    "What are the main components that enabled the deep learning breakthrough in the past decade?",
    
    # INTERMEDIATE - Deep Learning Mechanisms (3 questions)
    "How does backpropagation work to train neural networks?",
    "What is transfer learning and how is it applied in computer vision applications?",
    "How do convolutional neural networks process images using spatial invariance?",
    
    # ADVANCED - Applications to Self-Driving Cars (2 questions)
    "What are the main challenges and limitations of applying deep learning to autonomous vehicles?",
    "Why do most successful autonomous vehicle systems today rely more on non-machine learning methods than deep learning?",
    
    # ADVANCED - Deep Learning Limitations & Ethics (2 questions)
    "What is the overfitting problem and what regularization techniques can prevent it?",
    "What are the ethical concerns and unintended consequences of deep learning systems, and how does the coast runners game example illustrate this?"
]

# Ground truth answers based on the MIT 6.S094 lecture transcript
GROUND_TRUTH_ANSWERS = {
    "What is the 6.S094 course about and what will students learn?": 
        "6.S094 Deep Learning for Self-Driving Cars is part of MIT's curriculum designed to be accessible and provide the big picture of deep learning as it applies to building intelligent systems, particularly self-driving cars. The course teaches how to teach a car to drive itself and how to build an autopilot system. The methods taught are applicable to broader problems in robotics, machine learning, perception, and beyond.",
    
    "What is deep learning and how does it automate pattern extraction from data?": 
        "Deep learning is a way to extract useful patterns from data in an automated way with as little human effort involved as possible. It works through the optimization of neural networks to form higher and higher levels of abstractions and representations in data. Deep learning automates feature extraction from raw data, removing the need for human experts to manually extract features as was done in the 80s and 90s.",
    
    "What are the main components that enabled the deep learning breakthrough in the past decade?": 
        "The breakthrough was enabled by: (1) Digitization of data - ability to access data easily in distributed fashion across the world, (2) Hardware - compute power from Moore's Law, GPUs, and ASICs like Google's TPUs, (3) Community - people working together globally through platforms like GitHub, (4) Tooling - libraries like TensorFlow and PyTorch that enable rapid development with higher levels of abstraction, allowing people to go from idea to solution faster with less knowledge required.",
    
    "How does backpropagation work to train neural networks?": 
        "Backpropagation works in two passes. The forward pass propagates input through the network to get predictions and computes error using a loss function. The backward pass flows the error backwards through the network and computes gradients to determine how each weight contributed to the error. Based on these gradients, an optimization algorithm combined with a learning rate adjusts the weights - increasing weights responsible for correct outputs and decreasing weights responsible for incorrect outputs.",
    
    "What is transfer learning and how is it applied in computer vision applications?": 
        "Transfer learning involves taking a pre-trained neural network like ResNet that was trained on ImageNet, chopping off the fully connected layers or some parts of the network, and then retraining it on a new specialized dataset. For example, to build a pedestrian detector, you take ResNet trained on general vision tasks and retrain it on your pedestrian dataset. Depending on dataset size, some previous layers are frozen while others are retrained. This is extremely effective in computer vision, audio, speech, and NLP.",
    
    "How do convolutional neural networks process images using spatial invariance?": 
        "CNNs use convolution filters that slide over the image, taking advantage of the spatial invariance of visual information. This means a cat in the top-left corner has the same features as a cat in the top-right corner. Instead of treating each image position independently with equal weight, CNNs slide learned filters across the image to detect features anywhere in the scene. Stacked feature layers of these convolution filters can form high-level abstractions of visual information.",
    
    "What are the main challenges and limitations of applying deep learning to autonomous vehicles?": 
        "Major challenges include: (1) The gap between image classification and true scene understanding - classification is very far from understanding, (2) Vulnerability to adversarial examples - small noise can fool systems, (3) Difficulty handling lighting variations, pose variations, and inter-class variation that humans handle easily, (4) Inability to understand physics, 3D structure, mental models, and what's in other people's minds, (5) The problem formulation - most autonomous vehicle systems are not formulated as data-driven learning but as model-based optimization methods, (6) Need for massive amounts of labeled data and expensive human annotation.",
    
    "Why do most successful autonomous vehicle systems today rely more on non-machine learning methods than deep learning?": 
        "According to the lecture, the 10 million miles that autonomous vehicles have achieved has been attributed mostly to non-machine learning methods. While deep learning is used for some perception tasks, particularly enhanced perception from visual texture information, most of the system doesn't involve extensive machine learning. Some recent work uses recurrent neural networks to predict future intent of different players in the scene, but these are very early steps. The majority of successful autonomous vehicle systems use model-based optimization methods rather than data-driven learning.",
    
    "What is the overfitting problem and what regularization techniques can prevent it?": 
        "Overfitting occurs when a model trains too well on the training dataset and memorizes it to the extent that it only does well on that specific data but doesn't generalize to future unseen data. This is especially problematic for small datasets. Prevention techniques include: (1) Early stopping - using a validation set to monitor when training error decreases but test error increases, (2) Dropout - randomly removing nodes and their edges during training with certain probability, (3) Regularization methods, (4) Batch normalization and its variants like batch renormalization, (5) Data augmentation - growing small datasets by cropping, stretching, shifting images.",
    
    "What are the ethical concerns and unintended consequences of deep learning systems, and how does the coast runners game example illustrate this?": 
        "When an algorithm learns from data based on an objective function, the consequences aren't always obvious. The coast runners game example shows this clearly: the game is a boat racing game where you get points for finishing time, finishing position, and collecting green turbos. The optimal human strategy is to race around the track. However, a reinforcement learning agent discovered the optimal strategy was to ignore racing entirely and just circle around collecting regenerating turbos by slamming into walls. This well-reasoned objective function had totally unexpected consequences. This shows the need for AI safety and keeping humans in the loop of machine learning to consider consequences ahead of time and ask the right questions about what answers mean."
}


def get_test_dataset(video_id: str = "kY14KfZQ1TI") -> List[Dict]:
    """
    Get enhanced test dataset for evaluation
    
    Args:
        video_id: Video ID to evaluate (MIT 6.S094)
    
    Returns:
        List of test cases with questions and ground truth
    
    Example:
        >>> dataset = get_test_dataset()
        >>> print(f"Dataset has {len(dataset)} test cases")
        >>> print(f"First question: {dataset[0]['question']}")
    """
    dataset = []
    
    for question in TEST_QUESTIONS:
        test_case = {
            "question": question,
            "video_id": video_id,
            "ground_truth": GROUND_TRUTH_ANSWERS.get(question, "")
        }
        dataset.append(test_case)
    
    return dataset


def print_dataset_summary():
    """Print detailed summary of test dataset"""
    dataset = get_test_dataset()
    
    n = len(dataset)
    n_gt = len([d for d in dataset if d["ground_truth"]])
    print("=" * 80)
    print(f"EVALUATION DATASET - {n} QUESTIONS")
    print("=" * 80)
    print(f"\nCourse: Deep Learning for Self-Driving Cars")
    print(f"Instructor: Lex Fridman")
    print(f"Institution: MIT")
    print(f"Video ID: kY14KfZQ1TI")
    print(f"\nTotal test cases: {n}")
    print(f"With ground truth: {n_gt}")
    print(f"\nEstimated evaluation time: ~8-12 minutes")
    print(f"Estimated cost: ~$0.30-0.50")
    
    print(f"\n{'='*80}")
    print("QUESTION BREAKDOWN BY CATEGORY:")
    print(f"{'='*80}")
    
    categories = [
        ("BASIC - Course Overview & Definitions", [TEST_QUESTIONS[0], TEST_QUESTIONS[1]]),
        ("INTERMEDIATE - Deep Learning Mechanisms", [TEST_QUESTIONS[2], TEST_QUESTIONS[3]]),
        ("ADVANCED - Self-Driving Car Applications", [TEST_QUESTIONS[4]])
    ]
    
    for i, (category, questions) in enumerate(categories, 1):
        print(f"\n{category}:")
        for j, q in enumerate(questions, 1):
            print(f"  {len([q for cat_name, cat_q in categories[:i-1] for q in cat_q]) + j}. {q}")
    
    print("\n" + "=" * 80)
    print("GROUND TRUTH COVERAGE:")
    print("=" * 80)
    for i, item in enumerate(dataset, 1):
        has_gt = "✓" if item['ground_truth'] else "✗"
        gt_length = len(item['ground_truth']) if item['ground_truth'] else 0
        print(f"{has_gt} Q{i:2d}: {item['question'][:65]:65s} ({gt_length} chars)")
    
    print("\n" + "=" * 80)


def get_questions_by_topic(topic: str) -> List[str]:
    """
    Get questions filtered by topic
    
    Args:
        topic: 'fundamentals', 'mechanisms', 'self-driving', or 'ethics'
    
    Returns:
        List of questions on that topic
    """
    ranges = {
        'fundamentals': (0, 3),
        'mechanisms': (3, 6),
        'self-driving': (6, 8),
        'ethics': (8, 10)
    }
    
    start, end = ranges.get(topic.lower(), (0, 10))
    return TEST_QUESTIONS[start:end]


if __name__ == "__main__":
    print_dataset_summary()
    
    print("\n" + "=" * 80)
    print("SAMPLE QUESTION & GROUND TRUTH:")
    print("=" * 80)
    sample_q = TEST_QUESTIONS[6]  # Self-driving car question
    print(f"\nQuestion: {sample_q}")
    print(f"\nGround Truth Answer:")
    print(f"{GROUND_TRUTH_ANSWERS[sample_q]}")
    print("\n" + "=" * 80)