"""
CBT Analysis Utilities
Common helper functions for CBT cognitive bias analysis
"""

def parse_list_string(s):
    """문자열 형태의 리스트를 실제 리스트로 변환"""
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        # ['item1', 'item2'] 형태의 문자열을 파싱
        import ast
        try:
            return ast.literal_eval(s)
        except:
            # 단순 문자열인 경우 공백으로 분리
            return [item.strip().strip("'\"") for item in s.split(',') if item.strip()]
    return []


def calculate_accuracy(expected, actual):
    """정확도 계산 (Jaccard similarity)"""
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    
    expected_set = set(expected)
    actual_set = set(actual)
    
    intersection = len(expected_set & actual_set)
    union = len(expected_set | actual_set)
    
    return intersection / union if union > 0 else 0.0


def calculate_f1_score(expected, actual):
    """F1 스코어 계산"""
    if not expected and not actual:
        return 1.0, 1.0, 1.0  # precision, recall, f1
    
    if not expected:
        return 0.0, 1.0, 0.0  # 예상이 없으면 recall=1, precision=0
    
    if not actual:
        return 1.0, 0.0, 0.0  # 실제가 없으면 precision=1, recall=0
    
    expected_set = set(expected)
    actual_set = set(actual)
    
    true_positives = len(expected_set & actual_set)
    false_positives = len(actual_set - expected_set)
    false_negatives = len(expected_set - actual_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1 