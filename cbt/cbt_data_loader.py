"""
CBT Data Loader
Handles loading and processing of test data for CBT analysis
"""

import json
import os

def load_data():
    data_path = 'testdata.json'
    
    try:
        print("Loading testdata.json...")
        with open(data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test cases from testdata.json")
        return test_data
    except FileNotFoundError:
        print(f"Error: {data_path} not found!")
        return []
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return []


def select_test_case(test_data, case_index=0):
    """특정 테스트 케이스 선택"""
    if not test_data:
        print("No test data available")
        return None
    
    if case_index >= len(test_data):
        print(f"Case index {case_index} out of range. Using case 0.")
        case_index = 0
    
    case = test_data[case_index]
    print(f"Selected case {case_index} (ID: {case['id']})")
    
    return case


def filter_complete_cases(test_data):
    """완전한 데이터가 있는 케이스만 필터링"""
    complete_cases = []
    for case in test_data:
        if (case.get('core_belief') and 
            case.get('fine_grained') and 
            case.get('distortions')):
            complete_cases.append(case)
    return complete_cases


def save_data():
    """모든 필드가 있는 완전한 테스트 데이터를 testdata.json에 저장"""
    print("Creating complete test dataset...")
    print("="*50)
    
    # 전체 테스트 데이터 로드
    test_data = load_data()
    if not test_data:
        print("Failed to load test data")
        return
    
    # 완전한 데이터가 있는 케이스만 필터링
    complete_cases = filter_complete_cases(test_data)
    
    print(f"Total cases: {len(test_data)}")
    print(f"Complete cases (with all required fields): {len(complete_cases)}")
    print(f"Filtered out: {len(test_data) - len(complete_cases)} cases")
    
    # testdata.json에 저장
    output_file = 'testdata.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(complete_cases, f, indent=2, ensure_ascii=False)
        print(f"\nComplete test data saved to: {output_file}")
        print(f"Saved {len(complete_cases)} complete test cases")
        
        # 저장된 데이터의 샘플 정보 출력
        if complete_cases:
            print(f"\nSample case (ID: {complete_cases[0]['id']}):")
            sample_case = complete_cases[0]
            print(f"  - ori_text: {sample_case.get('ori_text', 'N/A')[:50]}...")
            print(f"  - thoughts: {sample_case.get('thoughts', 'N/A')[:50]}...")
            print(f"  - core_belief_major: {sample_case.get('core_belief_major', 'N/A')}")
            print(f"  - fine_grained: {sample_case.get('fine_grained', 'N/A')}")
            print(f"  - distortions: {sample_case.get('distortions', 'N/A')}")
            
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        return
    
    print("="*50)


def load_complete_test_data():
    """situation 필드가 포함된 완전한 테스트 데이터 로드"""
    data_path = 'testdata_complete.json'
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} complete test cases (with situation field)")
        return test_data
    except FileNotFoundError:
        print(f"{data_path} not found!")
        print("Trying testdata.json as fallback...")
        
        # Fallback to testdata.json
        try:
            with open('testdata.json', 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"⚠️  Loaded {len(test_data)} test cases (without situation field)")
            return test_data
        except FileNotFoundError:
            print("No test data files found!")
            return []
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return [] 