"""
Dataset-specific CBT Evaluation System
각 테스트 데이터셋(core_major, core_fine, distortions)에 대한 개별 평가
CBT-CD 형태의 테이블 출력 지원
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)  # Add current CBT directory to path

# Graph of Thoughts imports
try:
    from graph_of_thoughts import controller, language_models
    from graph_of_thoughts.operations import GraphOfOperations
    GOT_AVAILABLE = True
    print("Graph of Thoughts imported successfully")
except ImportError as e:
    print(f"Graph of Thoughts not available: {e}")
    GOT_AVAILABLE = False

# Local imports
from cbt_prompter import CBTPrompter
from cbt_parser import CBTParser
from cbt_utils import parse_list_string
from cbt_main import create_graph, CBTGenerate, CBTScore
from cbt_evaluator import calculate_f1_score

# Graph of Thoughts controller import
from graph_of_thoughts import controller

@dataclass
class DatasetConfig:
    """데이터셋 설정 정보"""
    name: str
    file_path: str
    target_field: str
    display_name: str
    description: str

# 데이터셋 설정
DATASETS = {
    'core_major': DatasetConfig(
        name='core_major',
        file_path='CBT-Bench/core/core_major_test.json',
        target_field='core_belief_major',
        display_name='핵심신념 (Core Beliefs)',
        description='Beck의 인지 모델에 따른 핵심신념 분석'
    ),
    'core_fine': DatasetConfig(
        name='core_fine', 
        file_path='CBT-Bench/fine/core_fine_test.json',
        target_field='fine_grained',
        display_name='세부신념 (Fine-grained Beliefs)',
        description='구체적이고 세분화된 인지적 신념 분석'
    ),
    'distortions': DatasetConfig(
        name='distortions',
        file_path='CBT-Bench/distortions/distortions_test.json', 
        target_field='distortions',
        display_name='인지왜곡 (Cognitive Distortions)',
        description='Burns의 인지왜곡 10가지 유형 분석'
    )
}

@dataclass
class ProgressState:
    """Simplified progress tracking"""
    start_time: float
    current_step: int
    total_steps: int
    api_calls: int

class TimedController(controller.Controller):
    """Timed controller with minimal dependencies"""
    def __init__(self, progress_tracker, logger, max_time_minutes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress_tracker
        self.logger = logger
        self.max_time = max_time_minutes * 60
        self.start_time = time.time()
        
    def run(self):
        # Override to add timeout protection and logging
        original_query = self.lm.query
        
        def timed_query(*args, **kwargs):
            self.progress.api_calls += 1
            if time.time() - self.start_time > self.max_time:
                self.logger.log_timeout()
                raise TimeoutError(f"Analysis exceeded {self.max_time/60} minutes")
            
            api_start = time.time()
            result = original_query(*args, **kwargs)
            api_time = time.time() - api_start
            
            self.progress.current_step += 1
            
            return result
        
        self.lm.query = timed_query
        super().run()

class DatasetEvaluator:
    """데이터셋별 CBT 평가 시스템"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Language model 설정 파일 경로
        """
        self.config_path = config_path or os.path.join(parent_dir, "config.json")
        self.results_dir = os.path.join(current_dir, "results")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """특정 데이터셋 로드"""
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
        
        config = DATASETS[dataset_name]
        file_path = os.path.join(parent_dir, config.file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} cases from {config.display_name}")
        return data
    
    def run_single_case_analysis(self, case: Dict, case_index: int, total_cases: int) -> Dict:
        """단일 케이스 CBT 분석 실행"""
        if not GOT_AVAILABLE:
            raise RuntimeError("Graph of Thoughts not available")
        
        # Initialize language model
        try:
            lm = language_models.ChatGPT(self.config_path, model_name="chatgpt")
        except Exception as e:
            raise RuntimeError(f"Language model initialization failed: {e}")
        
        # Create graph
        gop = create_graph()
        
        # Create controller with timeout protection
        class DatasetTimedController(TimedController):
            def __init__(self, max_time_minutes=10, *args, **kwargs):
                # Create a minimal progress tracker
                progress = ProgressState(time.time(), 0, 12, 0)
                
                # Create a minimal logger
                class MinimalLogger:
                    def log_step(self, *args, **kwargs): pass
                    def log_error(self, msg): print(f"ERROR: {msg}")
                    def log_timeout(self): print("TIMEOUT: Analysis exceeded time limit")
                    def log_final_results(self, *args, **kwargs): pass
                
                logger = MinimalLogger()
                super().__init__(progress, logger, max_time_minutes, *args, **kwargs)
        
        ctrl = DatasetTimedController(
            max_time_minutes=10,
            lm=lm,
            graph=gop, 
            prompter=CBTPrompter(),
            parser=CBTParser(),
            problem_parameters={
                "ori_text": case["ori_text"],
                "situation": case.get("situation", ""),
                "thoughts": case["thoughts"],
                "current": "",
                "method": "extract_gap"
            }
        )
        
        try:
            start_time = time.time()
            ctrl.run()
            execution_time = time.time() - start_time
            
            # Extract results from specific operations
            results = self._extract_results_from_graph(gop)
            
            return {
                'case_id': case.get('id', case_index),
                'execution_time': execution_time,
                'api_calls': ctrl.progress.api_calls,
                'analysis': results,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'case_id': case.get('id', case_index),
                'execution_time': 0,
                'api_calls': 0,
                'analysis': {},
                'status': 'failed',
                'error': str(e)
            }
    
    def _extract_results_from_graph(self, gop: GraphOfOperations) -> Dict:
        """GoT 그래프에서 결과 추출"""
        results = {
            'objective_fact': '',
            'subjective_thought': '',
            'core_beliefs': [],
            'fine_grained': [],
            'distortions': []
        }
        
        # Find specific operations and extract results
        gap_extract_op = None
        core_keep_op = None
        fine_keep_op = None
        dist_keep_op = None
        
        for operation in gop.operations:
            if hasattr(operation, 'method'):
                if operation.method == "extract_gap":
                    gap_extract_op = operation
                elif operation.method == "core_belief_keep":
                    core_keep_op = operation
                elif operation.method == "fine_grained_keep":
                    fine_keep_op = operation
                elif operation.method == "distortion_keep":
                    dist_keep_op = operation
        
        # Extract gap analysis
        if gap_extract_op:
            for thought in gap_extract_op.get_thoughts():
                state = thought.state
                if 'of' in state and state['of']:
                    results['objective_fact'] = state['of']
                if 'st' in state and state['st']:
                    results['subjective_thought'] = state['st']
                break
        
        # Extract core beliefs
        if core_keep_op:
            for thought in core_keep_op.get_thoughts():
                state = thought.state
                if 'core_beliefs' in state and state['core_beliefs']:
                    core_beliefs = state['core_beliefs']
                    if isinstance(core_beliefs, str):
                        try:
                            import ast
                            core_beliefs = ast.literal_eval(core_beliefs)
                        except:
                            core_beliefs = [core_beliefs]
                    results['core_beliefs'] = core_beliefs if isinstance(core_beliefs, list) else [core_beliefs]
                    break
        
        # Extract fine-grained
        if fine_keep_op:
            for thought in fine_keep_op.get_thoughts():
                state = thought.state
                if 'fine_grained' in state and state['fine_grained']:
                    fine_grained = state['fine_grained']
                    if isinstance(fine_grained, str):
                        try:
                            import ast
                            fine_grained = ast.literal_eval(fine_grained)
                        except:
                            fine_grained = [fine_grained]
                    results['fine_grained'] = fine_grained if isinstance(fine_grained, list) else [fine_grained]
                    break
        
        # Extract distortions
        if dist_keep_op:
            for thought in dist_keep_op.get_thoughts():
                state = thought.state
                if 'distortions' in state and state['distortions']:
                    distortions = state['distortions']
                    if isinstance(distortions, str):
                        try:
                            import ast
                            distortions = ast.literal_eval(distortions)
                        except:
                            distortions = [distortions]
                    results['distortions'] = distortions if isinstance(distortions, list) else [distortions]
                    break
        
        return results
    
    def evaluate_dataset(self, dataset_name: str, num_cases: Optional[int] = None, save_details: bool = True) -> Dict:
        """특정 데이터셋에 대한 전체 평가 실행"""
        print(f"\n🔬 Dataset Evaluation: {DATASETS[dataset_name].display_name}")
        print(f"📝 {DATASETS[dataset_name].description}")
        print("=" * 60)
        
        # Load dataset
        data = self.load_dataset(dataset_name)
        config = DATASETS[dataset_name]
        
        # Limit number of cases if specified
        if num_cases:
            data = data[:num_cases]
            print(f"📊 Evaluating {len(data)} cases (limited from full dataset)")
        else:
            print(f"📊 Evaluating {len(data)} cases (full dataset)")
        
        # Results storage
        all_results = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []
        
        failed_cases = 0
        start_time = time.time()
        
        print("\n🔄 Starting case-by-case analysis...")
        print("-" * 40)
        
        # Process each case
        for i, case in enumerate(data):
            case_index = i + 1
            case_id = case.get('id', f'case_{case_index}')
            
            print(f"[{case_index:3d}/{len(data):3d}] Processing case {case_id}...", end=" ")
            
            # Check if case has required field
            expected_values = case.get(config.target_field, [])
            if not expected_values:
                print("❌ SKIP (no expected values)")
                failed_cases += 1
                continue
            
            # Run analysis
            case_start = time.time()
            result = self.run_single_case_analysis(case, case_index, len(data))
            case_time = time.time() - case_start
            
            if result['status'] == 'failed':
                print(f"❌ FAILED ({case_time:.1f}s)")
                failed_cases += 1
                continue
            
            # Extract predicted values based on dataset type
            if dataset_name == 'core_major':
                predicted_values = result['analysis'].get('core_beliefs', [])
            elif dataset_name == 'core_fine':
                predicted_values = result['analysis'].get('fine_grained', [])
            elif dataset_name == 'distortions':
                predicted_values = result['analysis'].get('distortions', [])
            
            # Calculate metrics
            precision, recall, f1 = calculate_f1_score(expected_values, predicted_values)
            
            # Calculate accuracy (exact match)
            accuracy = 1.0 if set(expected_values) == set(predicted_values) else 0.0
            
            # Store metrics
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            
            # Store detailed result
            detailed_result = {
                'case_id': case_id,
                'case_index': case_index,
                'expected': expected_values,
                'predicted': predicted_values,
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                },
                'execution_time': case_time,
                'api_calls': result.get('api_calls', 0)
            }
            all_results.append(detailed_result)
            
            print(f"✅ SUCCESS ({case_time:.1f}s) F1={f1:.3f}")
        
        total_time = time.time() - start_time
        successful_cases = len(all_results)
        
        print("-" * 40)
        print(f"📈 Analysis completed: {successful_cases}/{len(data)} successful")
        
        # Calculate summary statistics
        if successful_cases > 0:
            summary_stats = {
                'precision': {
                    'mean': np.mean(precision_scores),
                    'std': np.std(precision_scores),
                    'min': np.min(precision_scores),
                    'max': np.max(precision_scores)
                },
                'recall': {
                    'mean': np.mean(recall_scores),
                    'std': np.std(recall_scores),
                    'min': np.min(recall_scores),
                    'max': np.max(recall_scores)
                },
                'f1': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'min': np.min(f1_scores),
                    'max': np.max(f1_scores)
                },
                'accuracy': {
                    'mean': np.mean(accuracy_scores),
                    'std': np.std(accuracy_scores),
                    'min': np.min(accuracy_scores),
                    'max': np.max(accuracy_scores)
                }
            }
        else:
            # All zeros if no successful cases
            summary_stats = {metric: {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0} 
                           for metric in ['precision', 'recall', 'f1', 'accuracy']}
        
        # Create final evaluation result
        evaluation_result = {
            'dataset_info': {
                'name': dataset_name,
                'display_name': config.display_name,
                'description': config.description,
                'target_field': config.target_field,
                'total_cases': len(data),
                'successful_cases': successful_cases,
                'failed_cases': failed_cases,
                'success_rate': successful_cases / len(data) * 100 if len(data) > 0 else 0
            },
            'performance_metrics': summary_stats,
            'execution_info': {
                'total_time': total_time,
                'avg_time_per_case': total_time / len(data) if len(data) > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': all_results if save_details else []
        }
        
        # Save results
        if save_details:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.results_dir, f"{dataset_name}_evaluation_{timestamp}.json")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Detailed results saved: {result_file}")
        
        return evaluation_result
    
    def print_cbt_cd_table(self, results: Dict[str, Dict]) -> None:
        """CBT-CD 형태의 테이블 출력"""
        print("\n" + "=" * 80)
        print("CBT-CD Evaluation Results".center(80))
        print("=" * 80)
        
        # Table header
        print(f"{'Dataset':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
        print("-" * 80)
        
        # Print results for each dataset
        for dataset_name, result in results.items():
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                dataset_display = DATASETS[dataset_name].display_name[:18]
                
                precision = metrics['precision']['mean']
                recall = metrics['recall']['mean']
                f1 = metrics['f1']['mean']
                accuracy = metrics['accuracy']['mean']
                
                print(f"{dataset_display:<20} {precision:<12.1f} {recall:<12.1f} {f1:<12.1f} {accuracy:<12.1f}")
        
        print("-" * 80)
        
        # Calculate overall averages
        if results:
            all_precision = [r['performance_metrics']['precision']['mean'] for r in results.values() if 'performance_metrics' in r]
            all_recall = [r['performance_metrics']['recall']['mean'] for r in results.values() if 'performance_metrics' in r]
            all_f1 = [r['performance_metrics']['f1']['mean'] for r in results.values() if 'performance_metrics' in r]
            all_accuracy = [r['performance_metrics']['accuracy']['mean'] for r in results.values() if 'performance_metrics' in r]
            
            if all_precision:
                avg_precision = np.mean(all_precision)
                avg_recall = np.mean(all_recall)
                avg_f1 = np.mean(all_f1)
                avg_accuracy = np.mean(all_accuracy)
                
                print(f"{'AVERAGE':<20} {avg_precision:<12.1f} {avg_recall:<12.1f} {avg_f1:<12.1f} {avg_accuracy:<12.1f}")
        
        print("=" * 80)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='CBT Dataset Evaluation Tool')
    parser.add_argument('--dataset', 
                       choices=['core_major', 'core_fine', 'distortions', 'all'],
                       default='all',
                       help='Dataset to evaluate (default: all)')
    parser.add_argument('--num-cases', 
                       type=int, 
                       help='Number of cases to evaluate (default: all cases)')
    parser.add_argument('--save-details', 
                       action='store_true', 
                       default=True,
                       help='Save detailed results to JSON files')
    parser.add_argument('--config', 
                       type=str,
                       help='Path to language model config file')
    
    args = parser.parse_args()
    
    print("🧠 CBT Dataset Evaluation Tool")
    print("🔗 Powered by Graph of Thoughts")
    print("📚 Based on Beck's Cognitive Model")
    
    # Initialize evaluator
    evaluator = DatasetEvaluator(config_path=args.config)
    
    # Determine which datasets to evaluate
    if args.dataset == 'all':
        datasets_to_eval = list(DATASETS.keys())
    else:
        datasets_to_eval = [args.dataset]
    
    print(f"\n📋 Will evaluate: {', '.join([DATASETS[d].display_name for d in datasets_to_eval])}")
    if args.num_cases:
        print(f"🔢 Limited to {args.num_cases} cases per dataset")
    
    # Run evaluations
    all_results = {}
    
    for dataset_name in datasets_to_eval:
        try:
            result = evaluator.evaluate_dataset(
                dataset_name=dataset_name,
                num_cases=args.num_cases,
                save_details=args.save_details
            )
            all_results[dataset_name] = result
            
        except Exception as e:
            print(f"❌ Failed to evaluate {dataset_name}: {e}")
            continue
    
    # Print summary table
    if all_results:
        evaluator.print_cbt_cd_table(all_results)
        
        # Save combined results
        if args.save_details:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_file = os.path.join(evaluator.results_dir, f"combined_evaluation_{timestamp}.json")
            
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📁 Combined results saved: {combined_file}")
    
    else:
        print("❌ No evaluations completed successfully")

if __name__ == "__main__":
    main() 