"""
CBT Main - Optimized Version
Efficient CBT cognitive bias analysis using simplified Graph of Thoughts
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path for Graph of Thoughts
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Graph of Thoughts imports
try:
    from graph_of_thoughts import controller, language_models
    from graph_of_thoughts.operations import Generate, Score, KeepBestN
    from graph_of_thoughts.operations import GraphOfOperations
    from graph_of_thoughts.language_models import AbstractLanguageModel
    from graph_of_thoughts.prompter import Prompter
    from graph_of_thoughts.parser import Parser
    GOT_AVAILABLE = True
    print("Graph of Thoughts imported successfully")
except ImportError as e:
    print(f"Graph of Thoughts not available: {e}")
    GOT_AVAILABLE = False

# Local imports
from cbt_prompter import CBTPrompter
from cbt_parser import CBTParser
from cbt_utils import parse_list_string
from cbt_data_loader import load_data, select_test_case
from cbt_evaluator import run_full_evaluation, run_sample_evaluation
from cbt_score import CBTScore
from run import get_logger

# Custom CBT Score Operation
class CBTScore(Score):
    """Custom Score operation that passes method information to prompter"""
    
    def __init__(self, num_samples: int = 1, combined_scoring: bool = True):
        super().__init__(num_samples, combined_scoring)
        self.method = None  # Will be set in create_graph()
    
    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        """Override execute to pass method information to prompter"""
        previous_thoughts = self.get_previous_thoughts()

        assert len(self.predecessors) > 0, "Score operation needs at least one predecessor"

        # Pass method information to prompter through a temporary attribute
        original_method = getattr(prompter, '_current_method', None)
        prompter._current_method = self.method

        try:
            if self.combined_scoring:
                previous_thoughts_states = [thought.state for thought in previous_thoughts]
                prompt = prompter.score_prompt(previous_thoughts_states, **kwargs)
                self.logger.debug("Prompt for LM: %s", prompt)

                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_samples)
                )
                self.logger.debug("Responses from LM: %s", responses)
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
                
                for thought, score in zip(previous_thoughts, scores):
                    new_thought = thought.__class__.from_thought(thought)
                    new_thought.score = score
                    self.thoughts.append(new_thought)
            else:
                for thought in previous_thoughts:
                    new_thought = thought.__class__.from_thought(thought)
                    prompt = prompter.score_prompt([thought.state], **kwargs)
                    self.logger.debug("Prompt for LM: %s", prompt)

                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug("Responses from LM: %s", responses)
                    score = parser.parse_score_answer([thought.state], responses)[0]

                    new_thought.score = score
                    self.thoughts.append(new_thought)

            self.logger.info("Score operation %d scored %d thoughts", self.id, len(self.thoughts))
            
        finally:
            # Restore original method (cleanup)
            if original_method is not None:
                prompter._current_method = original_method
            else:
                if hasattr(prompter, '_current_method'):
                    delattr(prompter, '_current_method')

# Custom CBT Generate Operation
class CBTGenerate(Generate):
    """Custom Generate operation that passes method information to prompter"""
    
    def __init__(self, num_branches_prompt: int = 1, num_branches_response: int = 1):
        super().__init__(num_branches_prompt, num_branches_response)
        self.method = None  # Will be set in create_graph()
    
    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        """Override execute to pass method information to prompter"""
        previous_thoughts = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # no predecessors, use kwargs as base state
            previous_thoughts = [self.get_thought_class()(state=kwargs)]

        # Pass method information to prompter through a temporary attribute
        original_method = getattr(prompter, '_current_method', None)
        prompter._current_method = self.method

        try:
            for thought in previous_thoughts:
                base_state = thought.state
                # Add method information to the state for proper parsing
                base_state['method'] = self.method
                prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
                self.logger.debug("Prompt for LM: %s", prompt)
                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_branches_response)
                )
                self.logger.debug("Responses from LM: %s", responses)
                for new_state in parser.parse_generate_answer(base_state, responses):
                    new_state = {**base_state, **new_state}
                    # Keep method information in the new state
                    new_state['method'] = self.method
                    self.thoughts.append(self.get_thought_class()(new_state))
                    self.logger.debug(
                        "New thought %d created with state %s",
                        self.thoughts[-1].id,
                        self.thoughts[-1].state,
                    )
            if (
                len(self.thoughts)
                > self.num_branches_prompt
                * self.num_branches_response
                * len(previous_thoughts)
                and self.num_branches_prompt > 0
            ):
                self.logger.warning(
                    "Generate operation %d created more thoughts than expected",
                    self.id,
                )
            self.logger.info(
                "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
            )
            
        finally:
            # Restore original method (cleanup)
            if original_method is not None:
                prompter._current_method = original_method
            else:
                if hasattr(prompter, '_current_method'):
                    delattr(prompter, '_current_method')
    
    def get_thought_class(self):
        """Get the Thought class to use for creating new thoughts"""
        from graph_of_thoughts.operations import Thought
        return Thought

# Get the parent directory to access config
config_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ProgressState:
    """Simplified progress tracking"""
    start_time: float
    current_step: int
    total_steps: int
    api_calls: int

def create_graph():
    """
    Create GoT structure for CBT analysis following proper GoT theory
    """
    gop = GraphOfOperations()
    
    # GoT theory: Generate multiple candidates, Score thoroughly, Keep best
    # CBT pathway: Gap -> Core -> Fine -> Distortion
    
    operations = []
    
    # 1. Gap extraction (single, deterministic)
    gap_extract = CBTGenerate(1, 1)  # Use custom CBTGenerate instead of Generate
    gap_extract.method = "extract_gap"
    operations.append(gap_extract)
    
    # 2. Core belief analysis (3 candidates -> best 1)
    core_gen = CBTGenerate(1, 3)  # 5개에서 3개로 줄임
    core_gen.method = "core_belief"
    core_score = CBTScore(1, False)
    core_score.method = "core_belief_score"
    core_keep = KeepBestN(1, True)  # 2개에서 1개로 줄임
    core_keep.method = "core_belief_keep"
    operations.extend([core_gen, core_score, core_keep])
    
    # 3. Fine-grained analysis (3 candidates -> best 1)
    fine_gen = CBTGenerate(1, 3)  # 2, 5에서 1, 3으로 줄임  
    fine_gen.method = "fine_grained"
    fine_score = CBTScore(1, False)
    fine_score.method = "fine_grained_score"
    fine_keep = KeepBestN(1, True)  # 2개에서 1개로 줄임
    fine_keep.method = "fine_grained_keep"
    operations.extend([fine_gen, fine_score, fine_keep])
    
    # 4. Distortion analysis (3 candidates -> best 1)
    dist_gen = CBTGenerate(1, 3)  # 2, 4에서 1, 3으로 줄임
    dist_gen.method = "distortion"
    dist_score = CBTScore(1, False)
    dist_score.method = "distortion_score"
    dist_keep = KeepBestN(1, True)
    dist_keep.method = "distortion_keep"
    operations.extend([dist_gen, dist_score, dist_keep])
    
    # 5. Final integration
    final_gen = CBTGenerate(1, 1)  # Use custom CBTGenerate instead of Generate
    final_gen.method = "final_integration"
    final_score = CBTScore(1, False)  # Use custom CBTScore instead of Score
    final_score.method = "final"
    operations.extend([final_gen, final_score])
    
    # Add all operations
    for op in operations:
        gop.add_operation(op)
    
    # Connect operations
    gap_extract.add_successor(core_gen)
    core_gen.add_successor(core_score)
    core_score.add_successor(core_keep)
    
    core_keep.add_successor(fine_gen)
    fine_gen.add_successor(fine_score)
    fine_score.add_successor(fine_keep)
    
    fine_keep.add_successor(dist_gen)
    dist_gen.add_successor(dist_score)
    dist_score.add_successor(dist_keep)
    
    dist_keep.add_successor(final_gen)
    final_gen.add_successor(final_score)
    
    return gop

def test_graph_structure():
    """Test optimized graph structure"""
    print("Testing optimized GoT graph structure...")
    gop = create_graph()
    
    print(f"Graph created with {len(gop.operations)} operations")
    
    # Verify no cycles
    visited = set()
    def has_cycle(op, path):
        if op in path:
            return True
        if op in visited:
            return False
        visited.add(op)
        path.add(op)
        for successor in op.successors:
            if has_cycle(successor, path):
                return True
        path.remove(op)
        return False
    
    # Check for cycles starting from each operation
    for op in gop.operations:
        if has_cycle(op, set()):
            print("ERROR: Cycle detected in graph!")
            return False
    
    print("Graph structure verified: No cycles detected")
    return True

def run_single_analysis(case_index=0):
    """
    Run single case CBT analysis using GoT
    """
    if not GOT_AVAILABLE:
        print("Graph of Thoughts not available")
        return None
    
    # Load test data
    test_data = load_data()
    if not test_data:
        print("No test data available")
        return None
    
    case_data = select_test_case(test_data, case_index)
    if not case_data:
        print("Failed to select test case")
        return None
    
    # Test graph structure first
    if not test_graph_structure():
        print("Graph structure test failed")
        return None
    
    print(f"Analyzing case {case_index}: {case_data['ori_text'][:100]}...")
    
    # Initialize real-time logger
    logger = get_logger("results")
    session_id = logger.start_session(
        case_id=case_data.get('id', case_index),
        case_data={
            'ori_text': case_data['ori_text'],
            'thoughts': case_data['thoughts'],
            'situation': case_data.get('situation', ''),
            'expected_results': {
                'core_belief': case_data.get('core_belief', []),
                'fine_grained': case_data.get('fine_grained', []),
                'distortions': case_data.get('distortions', [])
            }
        }
    )
    
    start_time = time.time()
    progress = ProgressState(start_time, 0, 12, 0)
    
    # Create optimized graph
    gop = create_graph()
    
    # Initialize language model
    try:
        config_path = os.path.join(config_parent_dir, "config.json")
        lm = language_models.ChatGPT(config_path, model_name="chatgpt")
    except Exception as e:
        error_msg = f"Language model initialization failed: {e}"
        print(error_msg)
        logger.log_error(error_msg)
        return None
    
    # Timed controller with real-time logging
    class TimedController(controller.Controller):
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
                elapsed = time.time() - self.progress.start_time
                print(f"Step {self.progress.current_step}/{self.progress.total_steps} | "
                      f"API: {api_time:.1f}s | Total: {elapsed:.1f}s")
                
                # Log each step with real-time logger
                step_name = getattr(self.prompter, '_current_method', 'unknown_step')
                input_data = {
                    'prompt': str(args[0])[:200] + "..." if args and len(str(args[0])) > 200 else str(args[0]) if args else "No input",
                    'method': step_name
                }
                output_data = {
                    'response': str(result)[:500] + "..." if len(str(result)) > 500 else str(result),
                    'response_length': len(str(result))
                }
                
                self.logger.log_step(
                    step_name=step_name,
                    step_number=self.progress.current_step,
                    input_data=input_data,
                    output_data=output_data,
                    duration_seconds=api_time,
                    api_calls_count=1
                )
                
                return result
            
            self.lm.query = timed_query
            super().run()
    
    # Create controller with timeout protection and logging
    ctrl = TimedController(
        progress_tracker=progress,
        logger=logger,
        max_time_minutes=10,
        lm=lm,
        graph=gop,
        prompter=CBTPrompter(),
        parser=CBTParser(),
        problem_parameters={
            "ori_text": case_data["ori_text"],
            "situation": case_data.get("situation", ""),
            "thoughts": case_data["thoughts"],
            "current": "",
            "method": "extract_gap"
        }
    )
    
    try:
        ctrl.run()
        
        # Extract final results - get only the BEST result from each final operation
        final_thoughts = ctrl.get_final_thoughts()
        if not final_thoughts:
            error_msg = "No analysis results generated"
            print(error_msg)
            logger.log_error(error_msg)
            return None
        
        # Get the best results from specific final operations only
        best_results = {
            'core_beliefs': [],
            'fine_grained': [],
            'distortions': [],
            'of': '',
            'st': ''
        }
        
        best_thought = None
        best_score = -1
        
        # Get results from specific final operations only (not all operations)
        core_keep_op = None
        fine_keep_op = None  
        dist_keep_op = None
        gap_extract_op = None
        
        # Find the specific operations we need
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
        
        # Extract gap analysis (of, st) from first operation
        if gap_extract_op:
            for thought in gap_extract_op.get_thoughts():
                state = thought.state
                if 'of' in state and state['of']:
                    best_results['of'] = state['of']
                if 'st' in state and state['st']:
                    best_results['st'] = state['st']
                break  # Take only the first (and only) result
        
        # Extract core beliefs from the KEEP operation (final selected result)
        if core_keep_op:
            for thought in core_keep_op.get_thoughts():
                state = thought.state
                if 'core_beliefs' in state and state['core_beliefs']:
                    core_beliefs = state['core_beliefs']
                    if isinstance(core_beliefs, str):
                        import ast
                        try:
                            core_beliefs = ast.literal_eval(core_beliefs)
                        except:
                            core_beliefs = [core_beliefs]
                    best_results['core_beliefs'] = core_beliefs if isinstance(core_beliefs, list) else [core_beliefs]
                    break  # Take only the best result
        
        # Extract fine-grained from the KEEP operation (final selected result)
        if fine_keep_op:
            for thought in fine_keep_op.get_thoughts():
                state = thought.state
                if 'fine_grained' in state and state['fine_grained']:
                    fine_grained = state['fine_grained']
                    if isinstance(fine_grained, str):
                        import ast
                        try:
                            fine_grained = ast.literal_eval(fine_grained)
                        except:
                            fine_grained = [fine_grained]
                    best_results['fine_grained'] = fine_grained if isinstance(fine_grained, list) else [fine_grained]
                    break  # Take only the best result
        
        # Extract distortions from the KEEP operation (final selected result)
        if dist_keep_op:
            for thought in dist_keep_op.get_thoughts():
                state = thought.state
                if 'distortions' in state and state['distortions']:
                    distortions = state['distortions']
                    if isinstance(distortions, str):
                        import ast
                        try:
                            distortions = ast.literal_eval(distortions)
                        except:
                            distortions = [distortions]
                    best_results['distortions'] = distortions if isinstance(distortions, list) else [distortions]
                    break  # Take only the best result
        
        # Find best thought for overall score
        for operation in gop.operations:
            for thought in operation.get_thoughts():
                if hasattr(thought, 'score') and thought.score > best_score:
                    best_score = thought.score
                    best_thought = thought
        
        # Parse results using cbt_utils
        actual_core = parse_list_string(best_results['core_beliefs'])
        actual_fine = parse_list_string(best_results['fine_grained'])
        actual_dist = parse_list_string(best_results['distortions'])
        
        results = {
            'case_id': case_data.get('id', case_index),
            'execution_time': time.time() - start_time,
            'api_calls': progress.api_calls,
            'got_score': best_score,
            'analysis': {
                'objective_fact': best_results['of'],
                'subjective_thought': best_results['st'],
                'core_beliefs': actual_core,
                'fine_grained': actual_fine,
                'distortions': actual_dist
            }
        }
        
        # Log final results
        logger.log_final_results(
            final_results=results,
            accuracy_results=None
        )
        
        return results
        
    except TimeoutError as e:
        print(f"Analysis timeout: {e}")
        logger.log_timeout()
        return None
    except Exception as e:
        error_msg = f"Analysis failed: {e}"
        print(error_msg)
        logger.log_error(error_msg)
        return None

def run_batch_evaluation(num_cases=5):
    """Run evaluation on multiple cases"""
    results = []
    start_time = time.time()
    
    print(f"Starting batch evaluation of {num_cases} cases...")
    
    for i in range(num_cases):
        print(f"\n--- Case {i+1}/{num_cases} ---")
        result = run_single_analysis(case_index=i)
        if result:
            results.append(result)
        else:
            print(f"Case {i+1} failed")
    
    total_time = time.time() - start_time
    
    if results:
        # Calculate summary statistics
        avg_time = sum(r['execution_time'] for r in results) / len(results)
        avg_api_calls = sum(r['api_calls'] for r in results) / len(results)
        
        print(f"\nBATCH EVALUATION COMPLETE")
        print(f"Successful cases: {len(results)}/{num_cases}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per case: {avg_time:.1f}s")
        print(f"Average API calls: {avg_api_calls:.1f}")
    
    return results

def main():
    """Main entry point for CBT analysis"""
    parser = argparse.ArgumentParser(description='CBT Dataset Evaluation using Graph of Thoughts')
    parser.add_argument('--dataset', 
                       choices=['core_major', 'core_fine', 'distortions', 'all'],
                       default='all',
                       help='Dataset to evaluate (default: all)')
    parser.add_argument('--num-cases', 
                       type=int, 
                       help='Number of cases to evaluate (default: all cases)')
    parser.add_argument('--mode',
                       choices=['single', 'batch', 'evaluation'],
                       default='evaluation',
                       help='Evaluation mode (default: evaluation)')
    
    args = parser.parse_args()
    
    print(f"🧠 CBT Analysis - Dataset: {args.dataset}")
    
    if args.mode == 'single':
        # Single case analysis
        result = run_single_analysis(case_index=0)
        if result:
            print("\nSingle case analysis completed successfully")
        else:
            print("Single case analysis failed")
            
    elif args.mode == 'batch':
        # Batch evaluation
        num_cases = args.num_cases or 5
        batch_results = run_batch_evaluation(num_cases=num_cases)
        
    elif args.mode == 'evaluation':
        # Full dataset evaluation using dataset_evaluator
        try:
            from dataset_evaluator import DatasetEvaluator
            
            print(f"🔬 Running evaluation on {args.dataset} dataset...")
            evaluator = DatasetEvaluator()
            
            if args.dataset == 'all':
                # Evaluate all datasets
                all_results = {}
                for dataset_name in ['core_major', 'core_fine', 'distortions']:
                    try:
                        result = evaluator.evaluate_dataset(
                            dataset_name=dataset_name,
                            num_cases=args.num_cases,
                            save_details=True
                        )
                        all_results[dataset_name] = result
                    except Exception as e:
                        print(f"Failed to evaluate {dataset_name}: {e}")
                        continue
                
                # Print summary table
                if all_results:
                    evaluator.print_cbt_cd_table(all_results)
                else:
                    print("No evaluations completed successfully")
            else:
                # Evaluate single dataset
                result = evaluator.evaluate_dataset(
                    dataset_name=args.dataset,
                    num_cases=args.num_cases,
                    save_details=True
                )
                
                # Print single dataset result as table
                evaluator.print_cbt_cd_table({args.dataset: result})
                
        except ImportError as e:
            print(f"Dataset evaluator not available: {e}")
            print("Falling back to legacy evaluation...")
            # Fallback to original cbt_evaluator functions
            try:
                from cbt_evaluator import run_full_evaluation, run_sample_evaluation
                
                if args.num_cases:
                    run_sample_evaluation(num_samples=args.num_cases)
                else:
                    run_full_evaluation()
            except ImportError:
                print("No evaluation functions available")
                print("Using batch evaluation as final fallback...")
                num_cases = args.num_cases or 5
                batch_results = run_batch_evaluation(num_cases=num_cases)

if __name__ == "__main__":
    main() 