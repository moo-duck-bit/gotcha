"""
CBT Evaluator - Progress & Results
Handles evaluation of CBT analysis results with realtime JSON saving

Based on Graph of Thoughts: Solving Elaborate Problems with Large Language Models
by Besta et al. (2024) - AAAI Conference on Artificial Intelligence
DOI: https://doi.org/10.1609/aaai.v38i16.29720
"""

import json
import os
import sys
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional, Set

# Configure logging for better error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Graph of Thoughts imports
try:
    from graph_of_thoughts import controller, language_models
    from graph_of_thoughts.operations import Generate, Score, KeepBestN
    from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
    GOT_AVAILABLE = True
    logger.info("Graph of Thoughts library loaded successfully")
except ImportError as e:
    logger.error(f"Graph of Thoughts not available: {e}")
    GOT_AVAILABLE = False
    # Define dummy GraphOfOperations for type hints
    class GraphOfOperations:
        pass

# Local imports
try:
    from cbt_prompter import CBTPrompter
    from cbt_parser import CBTParser
    from cbt_utils import parse_list_string, calculate_f1_score
    from cbt_data_loader import load_complete_test_data
except ImportError as e:
    logger.error(f"Local imports failed: {e}")
    raise

# Get the parent directory to access config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
result_dir = os.path.join(parent_dir, "result")

# Ensure result directory exists
os.makedirs(result_dir, exist_ok=True)


class CycleDetectionError(Exception):
    """Custom exception for cycle detection in Graph of Thoughts"""
    pass


class CBTAnalysisError(Exception):
    """Custom exception for CBT analysis errors"""
    pass


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


def detect_cycles(gop: GraphOfOperations) -> bool:
    """
    Detect cycles in Graph of Operations using DFS-based approach
    
    Based on cycle detection algorithms described in:
    - Tarjan, R. (1972). "Depth-first search and linear graph algorithms"
    - Cormen et al. (2009). "Introduction to Algorithms, 3rd Edition"
    
    Args:
        gop: Graph of Operations to check for cycles
        
    Returns:
        bool: True if cycles detected, False otherwise
        
    Raises:
        CycleDetectionError: If cycle detection fails
    """
    try:
        visited: Set[int] = set()
        rec_stack: Set[int] = set()
        
        def dfs_cycle_check(node_id: int, adjacency_map: Dict) -> bool:
            """DFS-based cycle detection with recursion stack tracking"""
            if node_id in rec_stack:
                return True  # Back edge found - cycle detected
            if node_id in visited:
                return False  # Already processed
                
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Check all successors
            for successor_id in adjacency_map.get(node_id, []):
                if dfs_cycle_check(successor_id, adjacency_map):
                    return True
                    
            rec_stack.remove(node_id)
            return False
        
        # Build adjacency map from Graph of Operations
        operations = gop.operations
        node_to_id = {op: i for i, op in enumerate(operations)}
        adjacency_map = {}
        
        for i, op in enumerate(operations):
            successors = gop.get_successors(op)
            adjacency_map[i] = [node_to_id[succ] for succ in successors]
        
        # Check for cycles starting from each unvisited node
        for node_id in range(len(operations)):
            if node_id not in visited:
                if dfs_cycle_check(node_id, adjacency_map):
                    logger.warning(f"Cycle detected in Graph of Operations at node {node_id}")
                    return True
                    
        return False
        
    except Exception as e:
        raise CycleDetectionError(f"Cycle detection failed: {e}")


def create_evaluation_graph() -> GraphOfOperations:
    """
    Create CBT evaluation graph with cycle prevention
    
    Implementation follows Graph of Thoughts framework:
    - Sequential dependency between CBT stages (Beck's cognitive model)
    - Parallel processing within each stage
    - Controlled backtracking with cycle prevention
    
    Based on:
    - Beck, A.T. et al. (1979). "Cognitive therapy of depression" 
    - Besta, M. et al. (2024). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
    
    Returns:
        GraphOfOperations: Cycle-free CBT analysis graph
        
    Raises:
        CycleDetectionError: If cycles are detected in graph construction
        CBTAnalysisError: If graph creation fails
    """
    try:
        gop = GraphOfOperations()
        
        # 1단계: 초기 텍스트 분석 (Gap 추출)
        gap_extract = Generate(1, 3)
        
        # 2단계: 핵심 신념 분석 (단계 내 병렬처리)
        core_belief_gen = Generate(1, 5)
        core_belief_score = Score(1, False)
        core_belief_keep = KeepBestN(3, True)
        
        # 3단계: 세부 신념 분석 (단계 내 병렬처리, 이전 단계 의존)
        fine_grained_gen = Generate(1, 5)
        fine_grained_score = Score(1, False)
        fine_grained_keep = KeepBestN(3, True)
        
        # 4단계: 인지왜곡 분석 (단계 내 병렬처리, 이전 단계들 의존)
        distortion_gen = Generate(1, 5)
        distortion_score = Score(1, False)
        distortion_keep = KeepBestN(2, True)
        
        # 5단계: 결과 검증 (순환 방지)
        validation_gen = Generate(1, 3)
        validation_score = Score(1, False)
        
        # 6단계: 조건부 재분석 (제한된 backtracking)
        refinement_gen = Generate(1, 3)
        refinement_score = Score(1, False)
        
        # 7단계: 최종 통합
        final_integration = Generate(1, 2)
        final_score = Score(1, False)
        final_selection = KeepBestN(1, True)
        
        # 그래프에 연산 추가
        operations = [
            gap_extract,
            core_belief_gen, core_belief_score, core_belief_keep,
            fine_grained_gen, fine_grained_score, fine_grained_keep,
            distortion_gen, distortion_score, distortion_keep,
            validation_gen, validation_score,
            refinement_gen, refinement_score,
            final_integration, final_score, final_selection
        ]
        
        for op in operations:
            gop.add_operation(op)
        
        # 순차적 의존성 연결 (Beck's CBT model)
        # 1. Gap → Core Beliefs
        gap_extract.add_successor(core_belief_gen)
        core_belief_gen.add_successor(core_belief_score)
        core_belief_score.add_successor(core_belief_keep)
        
        # 2. Core Beliefs → Fine-grained
        core_belief_keep.add_successor(fine_grained_gen)
        fine_grained_gen.add_successor(fine_grained_score)
        fine_grained_score.add_successor(fine_grained_keep)
        
        # 3. Fine-grained → Distortions
        fine_grained_keep.add_successor(distortion_gen)
        distortion_gen.add_successor(distortion_score)
        distortion_score.add_successor(distortion_keep)
        
        # 4. 검증 단계
        distortion_keep.add_successor(validation_gen)
        validation_gen.add_successor(validation_score)
        
        # 5. 조건부 분기 (순환 방지)
        validation_score.add_successor(final_integration)  # Primary path
        
        # 6. 제한된 refinement (백트래킹 대신 forward path)
        validation_score.add_successor(refinement_gen)
        refinement_gen.add_successor(refinement_score)
        refinement_score.add_successor(final_integration)  # Forward to final
        
        # 7. 최종 통합
        final_integration.add_successor(final_score)
        final_score.add_successor(final_selection)
        
        # Cycle detection validation
        if detect_cycles(gop):
            raise CycleDetectionError("Cycles detected in CBT evaluation graph")
            
        logger.info("CBT evaluation graph created successfully with cycle prevention")
        return gop
        
    except Exception as e:
        logger.error(f"Failed to create evaluation graph: {e}")
        raise CBTAnalysisError(f"Graph creation failed: {e}")


def save_case_initial_info(timestamp: str, case_result: Dict) -> None:
    """
    Save initial case information with error handling
    
    Args:
        timestamp: Timestamp for file naming
        case_result: Case data to save
        
    Raises:
        IOError: If file writing fails
    """
    try:
        case_filename = f"case_{case_result['case_id']}_{timestamp}.json"
        case_filepath = os.path.join(result_dir, case_filename)
        
        with open(case_filepath, 'w', encoding='utf-8') as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Case file created: case_{case_result['case_id']}.json")
        
    except (IOError, OSError, json.JSONEncodeError) as e:
        logger.error(f"Failed to save case initial info: {e}")
        raise


def run_analysis(case: Dict, lm, gop: GraphOfOperations, case_id: str, 
                case_index: int, total_cases: int, timestamp: str) -> Dict:
    """
    Execute CBT analysis for individual case with comprehensive error handling
    
    Args:
        case: Test case data
        lm: Language model instance
        gop: Graph of Operations
        case_id: Case identifier
        case_index: Current case index
        total_cases: Total number of cases
        timestamp: Processing timestamp
        
    Returns:
        Dict: Analysis results
        
    Raises:
        CBTAnalysisError: If analysis fails
        ValueError: If input validation fails
    """
    try:
        # Input validation
        required_fields = ["ori_text", "thoughts"]
        for field in required_fields:
            if field not in case:
                raise ValueError(f"Missing required field: {field}")
        
        logger.info(f"Starting CBT analysis for case {case_index}/{total_cases} (ID: {case_id})")
        
        ori_text = case["ori_text"]
        situation = case.get("situation", "")
        thoughts = case["thoughts"]
        
        logger.debug(f"Text preview: {ori_text[:80]}...")
        logger.debug(f"Thoughts preview: {thoughts[:80]}...")
        
        # Create initial case result with validation
        initial_case_result = {
            'case_id': case_id,
            'case_index': case_index,
            'status': 'processing',
            'start_time': datetime.now().isoformat(),
            'ori_text': ori_text,
            'situation': situation,
            'thoughts': thoughts,
            'expected': {
                'core_beliefs': case.get('core_belief', []),
                'fine_grained': case.get('fine_grained', []),
                'distortions': case.get('distortions', [])
            },
            'actual': {},
            'scores': {},
            'processing_steps': []
        }
        
        # Save initial case info
        save_case_initial_info(timestamp, initial_case_result)
        
        logger.info("Graph Pipeline: Gap→Core(Gen→Score→Keep)→Fine(Gen→Score→Keep)→Dist(Gen→Score→Keep)")
        
        # Create controller with error handling
        try:
            ctrl = controller.Controller(
                lm=lm,
                graph=gop,
                prompter=CBTPrompter(),
                parser=CBTParser(),
                problem_parameters={
                    "ori_text": ori_text,
                    "situation": situation,
                    "thoughts": thoughts,
                    "current": "",
                    "method": "extract_gap"
                }
            )
        except Exception as e:
            raise CBTAnalysisError(f"Controller creation failed: {e}")
        
        logger.info("Starting Graph of Thoughts execution...")
        logger.info("Step 1: Gap Extraction: Separating objective facts from subjective thoughts...")
        
        # Execute GoT with timeout and error handling
        try:
            ctrl.run()
        except Exception as e:
            logger.error(f"GoT execution failed: {e}")
            raise CBTAnalysisError(f"Graph execution failed: {e}")
        
        logger.info("Step 2: Core Belief Analysis: Generate(5)->Score->KeepBest(3)")
        logger.info("Step 3: Fine-grained Analysis: Generate(5)->Score->KeepBest(3)")
        logger.info("Step 4: Distortion Classification: Generate(5)->Score->KeepBest(2)")
        
        # Process final results with validation
        final_thoughts = ctrl.get_final_thoughts()
        
        logger.info("Processing final results...")
        logger.info("CBT Assessment-Based Scoring System:")
        logger.info("- Beck Triangle Score: Evaluating core belief structure")
        logger.info("- Distortion Accuracy: Checking cognitive distortion identification")
        logger.info("- Consistency Score: Measuring internal consistency")
        
        # Parse results with error handling
        actual_results = {}
        if final_thoughts:
            best_thought = None
            best_score = -1
            
            logger.info(f"Found {len(final_thoughts)} thought groups from scoring system")
            
            for i, thought_group in enumerate(final_thoughts):
                logger.debug(f"Group {i+1}: {len(thought_group)} candidate thoughts")
                for j, thought in enumerate(thought_group):
                    if hasattr(thought, 'score'):
                        logger.debug(f"Candidate {j+1}: Assessment Score {thought.score:.3f}")
                        if thought.score > best_score:
                            best_score = thought.score
                            best_thought = thought
            
            if best_thought and hasattr(best_thought, 'state') and isinstance(best_thought.state, dict):
                state = best_thought.state
                actual_results = {
                    'core_beliefs': parse_list_string(state.get('core_beliefs', '')),
                    'fine_grained': parse_list_string(state.get('fine_grained', '')),
                    'distortions': parse_list_string(state.get('distortions', ''))
                }
                logger.info(f"Best candidate selected (Assessment Score: {best_score:.3f})")
                logger.info(f"Core Beliefs: {actual_results['core_beliefs']}")
                logger.info(f"Fine-grained: {actual_results['fine_grained']}")
                logger.info(f"Distortions: {actual_results['distortions']}")
            else:
                logger.warning("No valid analysis results found in best candidate")
        else:
            logger.warning("No analysis candidates generated")
        
        return actual_results
        
    except ValueError as e:
        logger.error(f"Input validation error for case {case_id}: {e}")
        raise
    except CBTAnalysisError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in case {case_id}: {e}")
        raise CBTAnalysisError(f"Analysis failed for case {case_id}: {e}")


def save_realtime_progress(timestamp, processed_cases, total_cases, current_result, all_results, summary_stats):
    """실시간으로 결과를 /result에 저장"""
    # 개별 케이스 결과 저장
    case_filename = f"case_{current_result['case_id']}_{timestamp}.json"
    case_filepath = os.path.join(result_dir, case_filename)
    
    with open(case_filepath, 'w', encoding='utf-8') as f:
        json.dump(current_result, f, indent=2, ensure_ascii=False)
    
    # 전체 진행상황 저장 (덮어쓰기)
    progress_filename = f"progress_{timestamp}.json"
    progress_filepath = os.path.join(result_dir, progress_filename)
    
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary_stats,
        'processed_cases': processed_cases,
        'total_cases': total_cases,
        'all_results': all_results
    }
    
    with open(progress_filepath, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved: case_{current_result['case_id']}.json & progress.json")


def initialize_evaluation_files(timestamp, total_cases):
    """평가 시작 시 초기 JSON 파일들 생성"""
    # 초기 진행상황 파일 생성
    progress_filename = f"progress_{timestamp}.json"
    progress_filepath = os.path.join(result_dir, progress_filename)
    
    initial_progress = {
        'timestamp': datetime.now().isoformat(),
        'status': 'started',
        'summary': {
            'processed': 0,
            'total': total_cases,
            'remaining': total_cases,
            'avg_f1': 0.0,
            'eta_minutes': 0.0,
            'core_f1': 0.0,
            'fine_f1': 0.0,
            'dist_f1': 0.0
        },
        'processed_cases': 0,
        'total_cases': total_cases,
        'all_results': []
    }
    
    with open(progress_filepath, 'w', encoding='utf-8') as f:
        json.dump(initial_progress, f, indent=2, ensure_ascii=False)
    
    print(f"Initialized: /result/progress_{timestamp}.json")
    return progress_filename


def run_full_evaluation():
    """전체 데이터셋에 대해 CBT 분석 및 F1 스코어 계산 - 실시간 저장"""
    print("CBT Full Evaluation with Realtime Saving")
    print("="*50)
    
    # 테스트 데이터 로드
    test_data = load_complete_test_data()
    if not test_data:
        print("Failed to load test data")
        return
    
    total_cases = len(test_data)
    print(f" Total cases: {total_cases}")
    
    if not GOT_AVAILABLE:
        print("Graph of Thoughts not available")
        return
    
    # Graph of Operations 생성
    gop = create_evaluation_graph()
    
    try:
        config_path = os.path.join(parent_dir, "config.json")
        lm = language_models.ChatGPT(config_path, model_name="chatgpt")
        print("Language model initialized")
    except Exception as e:
        print(f"Language model error: {e}")
        return
    
    # 결과 저장용
    all_results = []
    core_metrics = {'precision': [], 'recall': [], 'f1': []}
    fine_metrics = {'precision': [], 'recall': [], 'f1': []}
    dist_metrics = {'precision': [], 'recall': [], 'f1': []}
    
    processed_cases = 0
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f" Results saving to: /result/")
    
    # 🔥 초기 JSON 파일들 생성
    progress_filename = initialize_evaluation_files(timestamp, total_cases)
    
    print("="*50)
    print("🔄 Starting Case Processing...")
    print("="*50)
    
    # 각 테스트 케이스 처리
    for i, case in enumerate(test_data):
        case_id = case['id']
        case_index = i + 1
        
        # 필수 필드 확인
        if (not case.get('core_belief') or 
            not case.get('fine_grained') or 
            not case.get('distortions')):
            print(f"\n⏭️  [{case_index}/{total_cases}] Case ID {case_id}: SKIPPED (missing fields)")
            continue
        
        case_start_time = time.time()
        
        try:
            # CBT 분석 실행 (구체적인 진행상황 포함, 즉시 JSON 생성)
            actual_results = run_analysis(case, lm, gop, case_id, case_index, total_cases, timestamp)
            
            # 예상 결과
            expected_results = {
                'core_beliefs': case['core_belief'],
                'fine_grained': case['fine_grained'],
                'distortions': case['distortions']
            }
            
            print(f"\nExpected vs Actual Comparison:")
            print(f"   Expected Core: {expected_results['core_beliefs']}")
            print(f"   Expected Fine: {expected_results['fine_grained']}")
            print(f"   Expected Dist: {expected_results['distortions']}")
            
            # F1 스코어 계산
            core_p, core_r, core_f1 = calculate_f1_score(expected_results['core_beliefs'], actual_results.get('core_beliefs', []))
            fine_p, fine_r, fine_f1 = calculate_f1_score(expected_results['fine_grained'], actual_results.get('fine_grained', []))
            dist_p, dist_r, dist_f1 = calculate_f1_score(expected_results['distortions'], actual_results.get('distortions', []))
            
            case_processing_time = round(time.time() - case_start_time, 2)
            
            print(f"\n F1 Scores: Core={core_f1:.3f}, Fine={fine_f1:.3f}, Dist={dist_f1:.3f}")
            print(f"  Processing time: {case_processing_time}s")
            
            # 결과 저장
            case_result = {
                'case_id': case_id,
                'case_index': case_index,
                'status': 'completed',
                'ori_text': case['ori_text'],
                'situation': case.get('situation', ''),
                'thoughts': case['thoughts'],
                'expected': expected_results,
                'actual': actual_results,
                'scores': {
                    'core': {'precision': core_p, 'recall': core_r, 'f1': core_f1},
                    'fine': {'precision': fine_p, 'recall': fine_r, 'f1': fine_f1},
                    'dist': {'precision': dist_p, 'recall': dist_r, 'f1': dist_f1}
                },
                'processing_time': case_processing_time,
                'completion_time': datetime.now().isoformat()
            }
            
            all_results.append(case_result)
            
            # 메트릭 저장
            core_metrics['precision'].append(core_p)
            core_metrics['recall'].append(core_r)
            core_metrics['f1'].append(core_f1)
            fine_metrics['precision'].append(fine_p)
            fine_metrics['recall'].append(fine_r)
            fine_metrics['f1'].append(fine_f1)
            dist_metrics['precision'].append(dist_p)
            dist_metrics['recall'].append(dist_r)
            dist_metrics['f1'].append(dist_f1)
            
            processed_cases += 1
            
            # 진행상황 계산
            elapsed_time = time.time() - start_time
            avg_time_per_case = elapsed_time / processed_cases
            remaining_cases = total_cases - processed_cases
            eta_seconds = avg_time_per_case * remaining_cases
            eta_minutes = eta_seconds / 60
            
            # 현재 평균 F1 계산
            current_avg_f1 = (
                sum(core_metrics['f1']) / len(core_metrics['f1']) +
                sum(fine_metrics['f1']) / len(fine_metrics['f1']) +
                sum(dist_metrics['f1']) / len(dist_metrics['f1'])
            ) / 3
            
            # 요약 통계
            summary_stats = {
                'processed': processed_cases,
                'total': total_cases,
                'remaining': remaining_cases,
                'avg_f1': round(current_avg_f1, 3),
                'eta_minutes': round(eta_minutes, 1),
                'core_f1': round(sum(core_metrics['f1']) / len(core_metrics['f1']), 3),
                'fine_f1': round(sum(fine_metrics['f1']) / len(fine_metrics['f1']), 3),
                'dist_f1': round(sum(dist_metrics['f1']) / len(dist_metrics['f1']), 3)
            }
            
            # 🔥 실시간 저장 (매 케이스마다) - 완료된 정보로 업데이트
            save_realtime_progress(timestamp, processed_cases, total_cases, case_result, all_results, summary_stats)
            
            # 진행상황 요약 출력
            print(f"\n{'='*60}")
            print(f"CASE [{processed_cases}/{total_cases}] COMPLETED")
            print(f" Current Avg F1: {current_avg_f1:.3f} | Core: {summary_stats['core_f1']} | Fine: {summary_stats['fine_f1']} | Dist: {summary_stats['dist_f1']}")
            print(f"  ETA: {eta_minutes:.1f}m | Remaining: {remaining_cases} cases")
            print(f" Results saved to /result/")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n[{case_index}/{total_cases}] Case ID {case_id}: ERROR")
            print(f"   Error: {str(e)[:100]}...")
            continue
    
    # 최종 결과 계산 및 저장
    if processed_cases > 0:
        final_stats = {
            'total_cases': total_cases,
            'processed_cases': processed_cases,
            'core_f1': round(sum(core_metrics['f1']) / len(core_metrics['f1']), 3),
            'fine_f1': round(sum(fine_metrics['f1']) / len(fine_metrics['f1']), 3),
            'dist_f1': round(sum(dist_metrics['f1']) / len(dist_metrics['f1']), 3),
            'overall_f1': round((
                sum(core_metrics['f1']) / len(core_metrics['f1']) +
                sum(fine_metrics['f1']) / len(fine_metrics['f1']) +
                sum(dist_metrics['f1']) / len(dist_metrics['f1'])
            ) / 3, 3),
            'total_time_minutes': round((time.time() - start_time) / 60, 1)
        }
        
        # 최종 결과 저장
        final_filename = f"final_results_{timestamp}.json"
        final_filepath = os.path.join(result_dir, final_filename)
        
        final_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': final_stats,
            'detailed_results': all_results
        }
        
        with open(final_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("🏁 FINAL EVALUATION RESULTS")
        print("="*60)
        print(f" Processed: {processed_cases}/{total_cases}")
        print(f"🎯 Core F1: {final_stats['core_f1']}")
        print(f"🔍 Fine F1: {final_stats['fine_f1']}")
        print(f"🧠 Dist F1: {final_stats['dist_f1']}")
        print(f" Overall F1: {final_stats['overall_f1']}")
        print(f"  Total time: {final_stats['total_time_minutes']}m")
        print(f" Results saved to: /result/{final_filename}")
        print("="*60)


def run_sample_evaluation(num_samples=5):
    """샘플 데이터셋에 대해 CBT 분석 (테스트용) - JSON 저장 포함"""
    print(f"🧪 Sample Evaluation ({num_samples} cases)")
    
    # 테스트 데이터 로드
    test_data = load_complete_test_data()
    if not test_data:
        print("Failed to load test data")
        return
    
    sample_data = test_data[:num_samples]
    print(f" Processing {len(sample_data)} sample cases")
    
    if not GOT_AVAILABLE:
        print("Graph of Thoughts not available")
        return
    
    # Graph of Operations 생성
    gop = create_evaluation_graph()
    
    try:
        config_path = os.path.join(parent_dir, "config.json")
        lm = language_models.ChatGPT(config_path, model_name="chatgpt")
    except Exception as e:
        print(f"Language model error: {e}")
        return
    
    # 결과 저장용
    f1_scores = []
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f" Sample results saving to: /result/")
    
    # 🔥 초기 샘플 평가 파일 생성
    sample_progress_filename = f"sample_progress_{timestamp}.json"
    sample_progress_filepath = os.path.join(result_dir, sample_progress_filename)
    
    initial_sample_data = {
        'timestamp': datetime.now().isoformat(),
        'status': 'started',
        'type': 'sample_evaluation',
        'total_samples': len(sample_data),
        'processed_samples': 0,
        'results': []
    }
    
    with open(sample_progress_filepath, 'w', encoding='utf-8') as f:
        json.dump(initial_sample_data, f, indent=2, ensure_ascii=False)
    
    print(f" Initialized: /result/sample_progress_{timestamp}.json")
    
    print("="*50)
    print("🔄 Starting Sample Processing...")
    print("="*50)
    
    # 각 테스트 케이스 처리
    for i, case in enumerate(sample_data):
        case_id = case['id']
        case_index = i + 1
        
        try:
            # CBT 분석 실행 (구체적인 진행상황 포함, 즉시 JSON 생성)
            actual_results = run_analysis(case, lm, gop, case_id, case_index, len(sample_data), timestamp)
            
            # 예상 결과
            expected_results = {
                'core_beliefs': case['core_belief'],
                'fine_grained': case['fine_grained'],
                'distortions': case['distortions']
            }
            
            # F1 스코어 계산
            core_p, core_r, core_f1 = calculate_f1_score(expected_results['core_beliefs'], actual_results.get('core_beliefs', []))
            fine_p, fine_r, fine_f1 = calculate_f1_score(expected_results['fine_grained'], actual_results.get('fine_grained', []))
            dist_p, dist_r, dist_f1 = calculate_f1_score(expected_results['distortions'], actual_results.get('distortions', []))
            
            overall_f1 = (core_f1 + fine_f1 + dist_f1) / 3
            f1_scores.append(overall_f1)
            
            # 샘플 결과 저장
            sample_result = {
                'case_id': case_id,
                'case_index': case_index,
                'expected': expected_results,
                'actual': actual_results,
                'f1_scores': {
                    'core': core_f1,
                    'fine': fine_f1,
                    'dist': dist_f1,
                    'overall': overall_f1
                }
            }
            all_results.append(sample_result)
            
            # 샘플 진행상황 실시간 업데이트
            updated_sample_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'processing',
                'type': 'sample_evaluation',
                'total_samples': len(sample_data),
                'processed_samples': case_index,
                'current_avg_f1': round(sum(f1_scores) / len(f1_scores), 3),
                'results': all_results
            }
            
            with open(sample_progress_filepath, 'w', encoding='utf-8') as f:
                json.dump(updated_sample_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nSample [{case_index}/{len(sample_data)}] (ID {case_id}) | F1: {overall_f1:.3f}")
            print(f" Updated: sample_progress_{timestamp}.json")
            
        except Exception as e:
            print(f"\nSample [{case_index}/{len(sample_data)}] (ID {case_id}): Error")
            continue
    
    # 샘플 결과 요약
    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        
        # 최종 샘플 결과 저장
        final_sample_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'type': 'sample_evaluation',
            'total_samples': len(sample_data),
            'processed_samples': len(f1_scores),
            'final_avg_f1': round(avg_f1, 3),
            'results': all_results
        }
        
        with open(sample_progress_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Sample Average F1: {avg_f1:.3f}")
        print(f" Final results saved to: /result/sample_progress_{timestamp}.json")
    
    print("Sample evaluation completed")


def evaluate_final_results(expected, actual):
    """
    Final evaluation of CBT analysis results using accuracy and F1-score
    
    Args:
        expected: Expected results dictionary
        actual: Actual results dictionary
        
    Returns:
        Dict: Final evaluation metrics
    """
    try:
        eval_metrics = {}
        categories = ['core_beliefs', 'fine_grained', 'distortions']
        
        for category in categories:
            expected_list = expected.get(category, [])
            actual_list = actual.get(category, [])
            
            # Input validation
            if not isinstance(expected_list, list):
                logger.warning(f"Expected {category} is not a list, converting: {expected_list}")
                expected_list = []
            if not isinstance(actual_list, list):
                logger.warning(f"Actual {category} is not a list, converting: {actual_list}")
                actual_list = []
            
            # Calculate F1 score (returns precision, recall, f1)
            precision, recall, f1 = calculate_f1_score(expected_list, actual_list)
            eval_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            logger.debug(f"{category} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
        
        # Calculate overall metrics
        overall_f1 = sum(eval_metrics[cat]['f1_score'] for cat in categories) / len(categories)
        eval_metrics['overall'] = {
            'avg_f1_score': overall_f1
        }
        
        return eval_metrics
        
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        # Return zero metrics as fallback
        return {cat: {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0} 
                for cat in ['core_beliefs', 'fine_grained', 'distortions', 'overall']}


def save_final_result(case_result, timestamp):
    """
    Save final case result with comprehensive error handling
    
    Args:
        case_result: Complete case analysis result
        timestamp: Processing timestamp
        
    Raises:
        IOError: If file operations fail
    """
    try:
        # Individual case file with validation
        case_filename = f"case_{case_result['case_id']}_{timestamp}.json"
        case_filepath = os.path.join(result_dir, case_filename)
        
        with open(case_filepath, 'w', encoding='utf-8') as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final case result saved: {case_filename}")
        
        # Progress update to comprehensive summary
        summary_filename = f"evaluation_progress_{timestamp}.json"
        summary_filepath = os.path.join(result_dir, summary_filename)
        
        summary_data = {
            'timestamp': timestamp,
            'processed_cases': 1,
            'status': 'in_progress',
            'latest_case': case_result
        }
        
        # Update existing summary or create new
        if os.path.exists(summary_filepath):
            try:
                with open(summary_filepath, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
                    existing_summary['processed_cases'] += 1
                    existing_summary['latest_case'] = case_result
                    summary_data = existing_summary
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not read existing summary, creating new: {e}")
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    except (IOError, OSError, json.JSONEncodeError) as e:
        logger.error(f"Failed to save final result: {e}")
        raise


def run_evaluation():
    """
    Main evaluation function with comprehensive error handling
    
    Implementation follows:
    - Beck, A.T. et al. (1979). "Cognitive therapy of depression"
    - Besta, M. et al. (2024). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
    
    Raises:
        DataLoadError: If data loading fails
        CBTAnalysisError: If analysis configuration fails
        Exception: For other critical failures
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting CBT evaluation at {timestamp}")
    
    try:
        # Load test data with validation
        logger.info("Loading test data...")
        try:
            test_data = load_complete_test_data()
            if not test_data:
                raise DataLoadError("No test data loaded")
            logger.info(f"Loaded {len(test_data)} test cases")
        except Exception as e:
            raise DataLoadError(f"Failed to load test data: {e}")
        
        # Initialize language model with error handling
        logger.info("Initializing language model...")
        try:
            if not GOT_AVAILABLE:
                raise CBTAnalysisError("Graph of Thoughts library not available")
            
            lm = language_models.ChatGPT(
                config_path=os.path.join(parent_dir, "config.json"),
                model_name="gpt-4",
                cache=True
            )
            logger.info("Language model initialized successfully")
        except Exception as e:
            raise CBTAnalysisError(f"Language model initialization failed: {e}")
        
        # Create evaluation graph with cycle detection
        logger.info("Creating CBT evaluation graph...")
        try:
            gop = create_evaluation_graph()
            logger.info("Evaluation graph created with cycle detection")
        except (CycleDetectionError, CBTAnalysisError) as e:
            logger.error(f"Graph creation failed: {e}")
            raise
        
        # Process each case with comprehensive error handling
        total_cases = len(test_data)
        successful_cases = 0
        failed_cases = 0
        
        all_results = []
        
        logger.info(f"Starting evaluation of {total_cases} cases")
        logger.info("=" * 80)
        
        for case_index, case in enumerate(test_data, 1):
            case_id = case.get('case_id', f'case_{case_index}')
            
            try:
                logger.info(f"\n[{case_index}/{total_cases}] Processing case {case_id}")
                
                # Run individual case analysis
                actual_results = run_analysis(
                    case, lm, gop, case_id, case_index, total_cases, timestamp
                )
                
                # Calculate scores with error handling
                expected_results = {
                    'core_beliefs': case.get('core_belief', []),
                    'fine_grained': case.get('fine_grained', []),
                    'distortions': case.get('distortions', [])
                }
                
                scores = evaluate_final_results(expected_results, actual_results)
                
                # Create complete case result
                case_result = {
                    'case_id': case_id,
                    'case_index': case_index,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'expected': expected_results,
                    'actual': actual_results,
                    'scores': scores,
                    'processing_info': {
                        'graph_structure': 'Sequential CBT stages with parallel processing',
                        'cycle_detection': 'Enabled',
                        'beck_model_compliance': True
                    }
                }
                
                # Save individual result
                save_final_result(case_result, timestamp)
                all_results.append(case_result)
                successful_cases += 1
                
                logger.info(f"Case {case_id} completed successfully")
                logger.info(f"Scores - Core: {scores['core_beliefs']['f1_score']:.3f}, "
                           f"Fine: {scores['fine_grained']['f1_score']:.3f}, "
                           f"Distortions: {scores['distortions']['f1_score']:.3f}")
                
            except (ValueError, CBTAnalysisError) as e:
                failed_cases += 1
                logger.error(f"Case {case_id} failed: {e}")
                
                # Save error information
                error_result = {
                    'case_id': case_id,
                    'case_index': case_index,
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                
                try:
                    save_final_result(error_result, timestamp)
                except Exception as save_error:
                    logger.error(f"Failed to save error result: {save_error}")
                
                # Continue with next case instead of stopping
                continue
                
            except Exception as e:
                failed_cases += 1
                logger.error(f"Unexpected error in case {case_id}: {e}")
                continue
        
        # Generate final comprehensive summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        # Calculate overall statistics
        if successful_cases > 0:
            avg_scores = {
                'core_beliefs': sum(r['scores']['core_beliefs']['f1_score'] for r in all_results) / successful_cases,
                'fine_grained': sum(r['scores']['fine_grained']['f1_score'] for r in all_results) / successful_cases,
                'distortions': sum(r['scores']['distortions']['f1_score'] for r in all_results) / successful_cases
            }
            
            overall_avg = sum(avg_scores.values()) / len(avg_scores)
            
            logger.info(f"Total Cases: {total_cases}")
            logger.info(f"Successful: {successful_cases}")
            logger.info(f"Failed: {failed_cases}")
            logger.info(f"Success Rate: {successful_cases/total_cases*100:.1f}%")
            logger.info("\nAverage F1-Scores:")
            logger.info(f"  Core Beliefs: {avg_scores['core_beliefs']:.3f}")
            logger.info(f"  Fine-grained: {avg_scores['fine_grained']:.3f}")
            logger.info(f"  Distortions: {avg_scores['distortions']:.3f}")
            logger.info(f"  Overall: {overall_avg:.3f}")
            
            # Save comprehensive final summary
            final_summary = {
                'evaluation_metadata': {
                    'timestamp': timestamp,
                    'framework': 'Graph of Thoughts for CBT Analysis',
                    'reference': 'Besta et al. (2024) - AAAI',
                    'total_cases': total_cases,
                    'successful_cases': successful_cases,
                    'failed_cases': failed_cases,
                    'success_rate': successful_cases/total_cases*100
                },
                'average_scores': avg_scores,
                'overall_average': overall_avg,
                'theoretical_framework': {
                    'cognitive_model': 'Beck\'s Cognitive Triangle (1979)',
                    'graph_structure': 'Sequential dependency with parallel processing',
                    'cycle_prevention': 'DFS-based cycle detection (Tarjan 1972)'
                },
                'all_case_results': all_results
            }
            
            summary_path = os.path.join(result_dir, f"final_evaluation_summary_{timestamp}.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
                
            logger.info(f"\nFinal summary saved: final_evaluation_summary_{timestamp}.json")
        else:
            logger.error("No cases processed successfully")
            
    except (DataLoadError, CBTAnalysisError) as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected evaluation error: {e}")
        raise
    finally:
        logger.info(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    """
    Entry point with comprehensive error handling
    
    Academic references:
    - Beck, A.T., Rush, A.J., Shaw, B.F., & Emery, G. (1979). Cognitive therapy of depression.
    - Besta, M. et al. (2024). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. AAAI.
    - Tarjan, R. (1972). Depth-first search and linear graph algorithms. SIAM Journal on Computing.
    """
    try:
        logger.info("CBT Graph of Thoughts Evaluator")
        logger.info("Based on academic research:")
        logger.info("- Beck's Cognitive Therapy Model (1979)")
        logger.info("- Graph of Thoughts Framework (Besta et al., 2024)")
        logger.info("- Cycle Detection Algorithms (Tarjan, 1972)")
        
        run_evaluation()
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1) 