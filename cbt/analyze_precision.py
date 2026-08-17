"""
Precision 분석 및 개선 방안
"""

import json

def analyze_precision_issues():
    """precision 문제점 분석"""
    
    with open('distortion_eval_results_20250602_052645.json', 'r') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    
    print("=== Precision 문제점 분석 ===")
    
    # 1. 과분석 정도 분석
    pred_counts = [len(r['predicted']) for r in results]
    true_counts = [len(r['expected']) for r in results]
    
    print(f"평균 예측 개수: {sum(pred_counts)/len(pred_counts):.1f}")
    print(f"평균 실제 개수: {sum(true_counts)/len(true_counts):.1f}")
    print(f"과분석 비율: {(sum(pred_counts)/sum(true_counts)-1)*100:.1f}%")
    
    # 2. 가장 과분류된 사례들
    over_pred = [(r['id'], len(r['predicted']), len(r['expected']), r['predicted'], r['expected']) 
                 for r in results]
    over_pred.sort(key=lambda x: x[1]-x[2], reverse=True)
    
    print(f"\n가장 과분류된 사례들:")
    for i, (case_id, pred_count, true_count, predicted, expected) in enumerate(over_pred[:5]):
        print(f"{i+1}. ID {case_id}: {pred_count}개 예측 vs {true_count}개 실제")
        print(f"   예측: {predicted}")
        print(f"   실제: {expected}")
        
        # 잘못 예측한 것들
        false_positives = [p for p in predicted if p not in expected]
        print(f"   잘못 예측: {false_positives}")
        print()
    
    # 3. 자주 과분류되는 왜곡 유형
    from collections import Counter
    
    false_positive_counter = Counter()
    for r in results:
        expected_set = set(r['expected'])
        for pred in r['predicted']:
            if pred not in expected_set:
                false_positive_counter[pred] += 1
    
    print("자주 과분류되는 왜곡 유형:")
    for distortion, count in false_positive_counter.most_common():
        print(f"- {distortion}: {count}회")

if __name__ == "__main__":
    analyze_precision_issues() 