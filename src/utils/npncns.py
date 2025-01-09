import re

def extract_numbers(text):
    """Extract numbers from text"""
    return [float(num) for num in re.findall(r'\d+(?:\.\d+)?', text)]

def M(set1, set2):
    """Count common elements in two sets"""
    return len(set(set1) & set(set2))

def calculate_np(H_n, S_n):
    """Calculate Number Precision"""
    if len(H_n) == 0:
        return 0
    return M(H_n, S_n) / len(H_n)

def calculate_nc(D_n, H_n, S_n):
    """Calculate Number Coverage"""
    if len(S_n) == 0:
        return 0
    
    nr = M(H_n, S_n) / len(S_n)
    
    # Check if the intersection is empty
    if not set(D_n).isdisjoint(set(S_n)):
        return nr * len(S_n) / M(D_n, S_n)
    else:
        return 0

def calculate_ns(np, nc):
    """Calculate Number Selection"""
    if np + nc == 0:
        return 0
    return 2 * np * nc / (np + nc)
