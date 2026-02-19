import math

def repetition_rate(text):
    """Simple repetition rate metric."""
    words = text.split()
    unique_words = set(words)
    if len(words) == 0:
        return 0.0
    return 1 - len(unique_words) / len(words)

def perplexity(loss):
    """Convert cross-entropy loss to perplexity."""
    return math.exp(loss)