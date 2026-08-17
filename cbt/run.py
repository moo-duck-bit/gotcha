"""
CBT Real-time Result Logger
실시간으로 CBT 분석 결과를 저장하는 모듈
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class AnalysisStep:
    """분석 단계별 결과를 저장하는 데이터 클래스"""
    step_name: str
    step_number: int
    timestamp: str
    duration_seconds: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    api_calls_count: int
    score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class AnalysisSession:
    """전체 분석 세션 정보를 저장하는 데이터 클래스"""
    session_id: str
    case_id: Any
    start_time: str
    end_time: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    total_api_calls: int = 0
    case_data: Optional[Dict[str, Any]] = None
    steps: List[AnalysisStep] = None
    final_results: Optional[Dict[str, Any]] = None
    accuracy_results: Optional[Dict[str, Any]] = None
    status: str = "running"  # running, completed, failed, timeout
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []


class RealTimeLogger:
    """실시간 결과 저장을 담당하는 클래스"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Args:
            output_dir: 결과 파일들을 저장할 디렉토리
        """
        self.output_dir = output_dir
        self.current_session: Optional[AnalysisSession] = None
        self.result_file_path: Optional[str] = None
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def start_session(self, case_id: Any, case_data: Dict[str, Any]) -> str:
        """새로운 분석 세션 시작"""
        session_id = f"cbt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = AnalysisSession(
            session_id=session_id,
            case_id=case_id,
            start_time=datetime.now().isoformat(),
            case_data=case_data,
            steps=[],
            total_api_calls=0,
            status="running"
        )
        
        # 결과 파일 경로 설정
        self.result_file_path = os.path.join(
            self.output_dir, 
            f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # 초기 세션 정보 저장
        self._save_session()

        
        return session_id
    
    def log_step(self, 
                 step_name: str,
                 step_number: int,
                 input_data: Dict[str, Any],
                 output_data: Dict[str, Any],
                 duration_seconds: float,
                 api_calls_count: int = 1,
                 score: Optional[float] = None,
                 error: Optional[str] = None):
        """분석 단계별 결과 로깅"""
        
        if self.current_session is None:
            print("Warning: No active session. Call start_session() first.")
            return
        
        step = AnalysisStep(
            step_name=step_name,
            step_number=step_number,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration_seconds,
            input_data=input_data,
            output_data=output_data,
            api_calls_count=api_calls_count,
            score=score,
            error=error
        )
        
        # 세션에 단계 추가
        self.current_session.steps.append(step)
        self.current_session.total_api_calls += api_calls_count
        
        # 실시간 저장
        self._save_session()
        
        # 진행 상황 출력
        elapsed_total = time.time() - datetime.fromisoformat(self.current_session.start_time).timestamp()

        
        if error:
            print(f"   Error: {error}")
    
    def log_final_results(self, 
                         final_results: Dict[str, Any],
                         accuracy_results: Optional[Dict[str, Any]] = None):
        """최종 결과 로깅"""
        
        if self.current_session is None:
            print("Warning: No active session.")
            return
        
        self.current_session.final_results = final_results
        self.current_session.accuracy_results = accuracy_results
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.status = "completed"
        
        # 총 실행 시간 계산
        start_time = datetime.fromisoformat(self.current_session.start_time)
        end_time = datetime.fromisoformat(self.current_session.end_time)
        self.current_session.total_duration_seconds = (end_time - start_time).total_seconds()
        
        # 최종 저장
        self._save_session()
        
        print(f"Analysis completed!")
        print(f"Results saved to: {self.result_file_path}")
        print(f"Total duration: {self.current_session.total_duration_seconds:.1f}s")
        print(f"Total API calls: {self.current_session.total_api_calls}")
        
        if accuracy_results:
            overall_accuracy = accuracy_results.get('overall_accuracy')
            if overall_accuracy is not None and isinstance(overall_accuracy, (int, float)):
                print(f"Overall accuracy: {overall_accuracy:.2%}")
            else:
                print(f"Overall accuracy: {overall_accuracy}")
    
    def log_error(self, error_message: str):
        """에러 로깅"""
        
        if self.current_session is None:
            print("Warning: No active session.")
            return
        
        self.current_session.error_message = error_message
        self.current_session.status = "failed"
        self.current_session.end_time = datetime.now().isoformat()
        
        # 총 실행 시간 계산
        start_time = datetime.fromisoformat(self.current_session.start_time)
        end_time = datetime.fromisoformat(self.current_session.end_time)
        self.current_session.total_duration_seconds = (end_time - start_time).total_seconds()
        
        # 에러 상태로 저장
        self._save_session()
        
        print(f"Analysis failed: {error_message}")
        print(f"Error log saved to: {self.result_file_path}")
    
    def log_timeout(self):
        """타임아웃 로깅"""
        
        if self.current_session is None:
            return
        
        self.current_session.status = "timeout"
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.error_message = "Analysis timeout exceeded"
        
        # 총 실행 시간 계산
        start_time = datetime.fromisoformat(self.current_session.start_time)
        end_time = datetime.fromisoformat(self.current_session.end_time)
        self.current_session.total_duration_seconds = (end_time - start_time).total_seconds()
        
        self._save_session()
        
        print(f"Analysis timed out")
        print(f"Timeout log saved to: {self.result_file_path}")
    
    def _save_session(self):
        """현재 세션을 파일에 저장"""
        
        if self.current_session is None or self.result_file_path is None:
            return
        
        try:
            # 세션을 딕셔너리로 변환
            session_dict = asdict(self.current_session)
            
            # JSON 파일로 저장 (pretty print)
            with open(self.result_file_path, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save session: {e}")
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """현재 세션 정보 반환"""
        if self.current_session is None:
            return None
        
        return {
            'session_id': self.current_session.session_id,
            'case_id': self.current_session.case_id,
            'status': self.current_session.status,
            'steps_completed': len(self.current_session.steps),
            'total_api_calls': self.current_session.total_api_calls,
            'result_file': self.result_file_path
        }
    
    def create_summary_report(self) -> Dict[str, Any]:
        """분석 완료 후 요약 리포트 생성"""
        
        if self.current_session is None:
            return {}
        
        steps_summary = []
        total_step_time = 0
        
        for step in self.current_session.steps:
            step_summary = {
                'step_name': step.step_name,
                'step_number': step.step_number,
                'duration': step.duration_seconds,
                'api_calls': step.api_calls_count,
                'score': step.score,
                'has_error': step.error is not None
            }
            steps_summary.append(step_summary)
            total_step_time += step.duration_seconds
        
        summary = {
            'session_info': {
                'session_id': self.current_session.session_id,
                'case_id': self.current_session.case_id,
                'status': self.current_session.status,
                'total_duration': self.current_session.total_duration_seconds,
                'total_api_calls': self.current_session.total_api_calls
            },
            'performance_metrics': {
                'steps_completed': len(self.current_session.steps),
                'total_step_time': total_step_time,
                'avg_step_time': total_step_time / len(self.current_session.steps) if self.current_session.steps else 0,
                'avg_api_calls_per_step': self.current_session.total_api_calls / len(self.current_session.steps) if self.current_session.steps else 0
            },
            'steps_summary': steps_summary,
            'accuracy_results': self.current_session.accuracy_results,
            'final_results': self.current_session.final_results
        }
        
        return summary
    
    # 표준 로깅 메서드들 추가 (호환성을 위해)
    def debug(self, message: str, *args):
        """디버그 로깅"""
        print(f"DEBUG: {message}")
    
    def info(self, message: str, *args):
        """정보 로깅"""
        print(f"INFO: {message}")
    
    def warning(self, message: str, *args):
        """경고 로깅"""
        print(f"WARNING: {message}")
    
    def error(self, message: str, *args):
        """에러 로깅"""
        print(f"ERROR: {message}")
        if self.current_session:
            self.log_error(message)


# 전역 로거 인스턴스 (싱글톤 패턴)
_global_logger: Optional[RealTimeLogger] = None

def get_logger(output_dir: str = "results") -> RealTimeLogger:
    """전역 로거 인스턴스 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = RealTimeLogger(output_dir)
    return _global_logger

def reset_logger():
    """전역 로거 리셋 (테스트용)"""
    global _global_logger
    _global_logger = None


# =====================================================
# CBT 분석 실행 스크립트
# =====================================================

import sys
import argparse

def main():
    """CBT 분석 실행 메인 함수"""
    parser = argparse.ArgumentParser(description='CBT Analysis with Real-time Logging')
    parser.add_argument('--case', type=int, default=0, help='Case index to analyze (default: 0)')
    parser.add_argument('--batch', type=int, help='Run batch evaluation with N cases')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    print("CBT Analysis with Real-time Logging")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Reset logger to ensure clean state
    reset_logger()
    
    try:
        # Import CBT analysis functions
        from cbt_main import run_single_analysis, run_batch_evaluation
        
        if args.batch:
            print(f"Running batch evaluation with {args.batch} cases...")
            results = run_batch_evaluation(num_cases=args.batch)
            
            if results:
                print(f"\nBatch evaluation completed successfully!")
                print(f"Processed {len(results)} cases")
                
                # Create summary
                successful_cases = len(results)
                total_time = sum(r['execution_time'] for r in results)
                total_api_calls = sum(r['api_calls'] for r in results)
                
                print(f"\nSUMMARY:")
                print(f"- Successful cases: {successful_cases}/{args.batch}")
                print(f"- Total execution time: {total_time:.1f}s")
                print(f"- Average time per case: {total_time/successful_cases:.1f}s")
                print(f"- Total API calls: {total_api_calls}")
                print(f"- Average API calls per case: {total_api_calls/successful_cases:.1f}")
                
            else:
                print("Batch evaluation failed")
                return 1
                
        else:
            print(f"Running single case analysis (case {args.case})...")
            result = run_single_analysis(case_index=args.case)
            
            if result:
                print(f"\nSingle case analysis completed successfully!")
                print(f"Case ID: {result['case_id']}")
                print(f"Execution time: {result['execution_time']:.1f}s")
                print(f"API calls: {result['api_calls']}")
                print(f"GoT score: {result['got_score']:.3f}")
                
                # Display analysis results
                analysis = result['analysis']
                print(f"\nANALYSIS RESULTS:")
                print(f"- Core beliefs: {analysis['core_beliefs']}")
                print(f"- Fine-grained beliefs: {analysis['fine_grained']}")
                print(f"- Cognitive distortions: {analysis['distortions']}")
                
            else:
                print("Single case analysis failed")
                return 1
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Analysis completed. Check the results directory for detailed logs.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 