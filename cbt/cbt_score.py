"""
CBT GoT Scoring Functions - Definition-based with Frequency Analysis
Internal scoring for Graph of Thoughts operations based on CBT definitions and LLM evaluation
"""

from collections import Counter
from typing import List, Dict, Any

class CBTScore:
    """GoT internal scoring for CBT analysis stages - frequency analysis + LLM-based scoring"""
    
    # CBT 정의들 (prompter.py 기반)
    CORE_BELIEF_DEFINITIONS = {
        'helpless': 'being ineffective—in getting things done, self-protection, and/or measuring up to other',
        'unlovable': 'having personal qualities resulting in an inability to get or maintain love and intimacy from others',
        'worthless': 'being an immoral sinner or dangerous to others'
    }
    
    FINE_GRAINED_DEFINITIONS = {
        'helpless': [
            'I am incompetent', 'I am helpless', 'I am powerless, weak, vulnerable',
            'I am a victim', 'I am needy', 'I am trapped', 'I am out of control',
            'I am a failure, loser', 'I am defective'
        ],
        'unlovable': [
            'I am unlovable', 'I am unattractive', 'I am undesirable, unwanted',
            'I am bound to be rejected', 'I am bound to be abandoned', 'I am bound to be alone'
        ],
        'worthless': [
            'I am worthless, waste', 'I am immoral', 'I am bad - dangerous, toxic, evil'
        ]
    }
    
    DISTORTION_DEFINITIONS = {
        'all-or-nothing thinking': 'black-and-white, polarized, or dichotomous thinking. View situation in only two categories instead of on a continuum',
        'mind reading': 'believe you know what others are thinking, failing to consider other, more likely possibilities',
        'fortune-telling': 'predict the future negatively without considering other, more likely outcomes',
        'personalization': 'believe others are behaving negatively because of you, without considering more plausible explanations',
        'emotional reasoning': 'think something must be true because you "feel" it so strongly, ignoring evidence to the contrary',
        'overgeneralization': 'make a sweeping negative conclusion that goes far beyond the current situation',
        'labeling': 'put a fixed, global label on yourself or others without considering evidence might lead to less extreme conclusion',
        'should statements': 'have a precise, fixed idea of how you or others should behave, overestimate how bad it is when expectations not met',
        'magnification': 'unreasonably magnify the negative and/or minimize the positive when evaluating yourself, another person, or situation',
        'mental filter': 'pay undue attention to one negative detail instead of seeing the whole picture'
    }
    
    @staticmethod
    def analyze_frequency(thoughts: List[Any], category: str) -> Dict:
        """
        GoT 여러 후보들 중 가장 자주 나타나는 조합/개별 응답 분석
        
        Args:
            thoughts: List of thought objects from GoT
            category: 'core_beliefs', 'fine_grained', 'distortions'
            
        Returns:
            Dict: 빈도 분석 결과와 가장 많이 나온 응답
        """
        all_responses = []
        all_combinations = []
        
        for thought in thoughts:
            if hasattr(thought, 'state') and category in thought.state:
                response = thought.state[category]
                
                # 파싱
                if isinstance(response, str):
                    import ast
                    try:
                        response = ast.literal_eval(response)
                    except:
                        response = [response]
                
                if isinstance(response, list):
                    # 개별 응답들 수집
                    all_responses.extend(response)
                    
                    # 조합 수집 (리스트를 tuple로 변환해서 hashable하게)
                    if len(response) > 1:
                        combination = tuple(sorted(response))
                        all_combinations.append(combination)
                    # 개별 응답은 중복 추가하지 않음
        
        # 빈도 분석
        response_freq = Counter(all_responses)
        combination_freq = Counter(all_combinations)
        
        # 가장 많이 나온 것 결정
        if combination_freq:
            most_common_combination = combination_freq.most_common(1)[0]
            most_frequent = {
                'type': 'combination',
                'value': list(most_common_combination[0]),
                'frequency': most_common_combination[1],
                'total_candidates': len(thoughts)
            }
        elif response_freq:
            most_common_response = response_freq.most_common(1)[0]
            most_frequent = {
                'type': 'individual',
                'value': [most_common_response[0]],
                'frequency': most_common_response[1],
                'total_candidates': len(thoughts)
            }
        else:
            most_frequent = {
                'type': 'none',
                'value': [],
                'frequency': 0,
                'total_candidates': len(thoughts)
            }
        
        return {
            'most_frequent': most_frequent,
            'all_responses': list(response_freq.keys()),
            'response_frequencies': dict(response_freq),
            'combination_frequencies': dict(combination_freq)
        }
    
    @staticmethod
    def generate_llm_scoring_prompt(category: str, analysis_result: Dict, context: Dict = None) -> str:
        """
        LLM이 점수를 매기기 위한 프롬프트 생성
        
        Args:
            category: 'core_beliefs', 'fine_grained', 'distortions'
            analysis_result: 분석 결과 (빈도 분석 포함)
            context: 사용자 텍스트, 상황 등
            
        Returns:
            str: LLM 점수 매기기 프롬프트
        """
        if category == 'core_beliefs':
            definitions = CBTScore.CORE_BELIEF_DEFINITIONS
            prompt_base = f"""
<Instruction>
You are a CBT expert. Please rate the results of the core belief analysis on a scale of 1 to 10.
[Core Belief Definitions]
{definitions}

[Analysis Results]
Most frequent response: {analysis_result['most_frequent']['value']}
Frequency: {analysis_result['most_frequent']['frequency']}/{analysis_result['most_frequent']['total_candidates']}
All responses: {analysis_result['all_responses']}

[User Context]
Text: {context.get('text', '') if context else ''}
Situation: {context.get('situation', '') if context else ''}
Thoughts: {context.get('thoughts', '') if context else ''}

[Evaluation Criteria - 1-10 Scale]
1-3: Not aligned with definitions or irrelevant to context
4-6: Partially aligned with definitions but weak connection to context
7-8: Aligned with definitions and strongly connected to context
9-10: Perfectly aligned with definitions and strongly connected to context

Only provide the score as a number (e.g., 7):
</Instruction>
"""
            
        elif category == 'fine_grained':
            definitions = CBTScore.FINE_GRAINED_DEFINITIONS
            prompt_base = f"""
<Instruction>
You are a CBT expert. Please rate the results of the fine-grained belief analysis on a scale of 1 to 10.

[Fine-grained Belief Definitions]
{definitions}

[Analysis Results]
Most frequent response: {analysis_result['most_frequent']['value']}
Frequency: {analysis_result['most_frequent']['frequency']}/{analysis_result['most_frequent']['total_candidates']}
All responses: {analysis_result['all_responses']}

[Core Belief Context]
Core Beliefs: {context.get('core_beliefs', '') if context else ''}

[Evaluation Criteria - 1-10 Scale]
1-3: Not aligned with definitions or irrelevant to core beliefs
4-6: Partially aligned with definitions but weak connection to core beliefs
7-8: Aligned with definitions and strongly connected to core beliefs
9-10: Perfectly aligned with definitions and strongly connected to core beliefs

Only provide the score as a number (e.g., 8):
</Instruction>
"""
            
        elif category == 'distortions':
            definitions = CBTScore.DISTORTION_DEFINITIONS
            prompt_base = f"""
<Instruction>
You are a CBT expert. Please rate the results of the distortion analysis on a scale of 1 to 10.

[Distortion Definitions]
{definitions}

[Analysis Results]
Most frequent response: {analysis_result['most_frequent']['value']}
Frequency: {analysis_result['most_frequent']['frequency']}/{analysis_result['most_frequent']['total_candidates']}
All responses: {analysis_result['all_responses']}

[Overall Analysis Context]
Core Beliefs: {context.get('core_beliefs', '') if context else ''}
Fine-grained Beliefs: {context.get('fine_grained', '') if context else ''}
Text: {context.get('text', '') if context else ''}

[Evaluation Criteria - 1-10 Scale]
1-3: Not aligned with definitions or irrelevant to overall analysis
4-6: Partially aligned with definitions but weak connection to overall analysis
7-8: Aligned with definitions and strongly connected to overall analysis
9-10: Perfectly aligned with definitions and strongly connected to overall analysis

Only provide the score as a number (e.g., 9):
</Instruction>
"""
        
        return prompt_base
    
    @staticmethod
    def score_core_belief(thought):
        """
        개별 thought 점수 (backward compatibility용)
        실제로는 analyze_frequency + LLM 점수가 권장됨
        """
        state = thought.state
        core_beliefs = state.get('core_beliefs', [])
        
        if not core_beliefs:
            return 1.0
        
        # 간단한 정의 매칭 점수
        valid_cores = ['helpless', 'unlovable', 'worthless']
        
        if isinstance(core_beliefs, str):
            import ast
            try:
                core_beliefs = ast.literal_eval(core_beliefs)
            except:
                core_beliefs = [core_beliefs]
        
        if not isinstance(core_beliefs, list):
            core_beliefs = [core_beliefs]
        
        matched = any(valid_core in str(belief).lower() for belief in core_beliefs for valid_core in valid_cores)
        
        return 7.0 if matched else 3.0  # 간단한 매칭 점수
    
    @staticmethod
    def score_fine_grained(thought):
        """개별 thought 점수 (backward compatibility용)"""
        state = thought.state
        fine_grained = state.get('fine_grained', [])
        
        if not fine_grained:
            return 1.0
        
        # 간단한 정의 매칭
        all_defined = []
        for beliefs in CBTScore.FINE_GRAINED_DEFINITIONS.values():
            all_defined.extend(beliefs)
        
        if isinstance(fine_grained, str):
            import ast
            try:
                fine_grained = ast.literal_eval(fine_grained)
            except:
                fine_grained = [fine_grained]
        
        matched = any(defined.lower() in str(belief).lower() for belief in fine_grained for defined in all_defined)
        
        return 7.0 if matched else 3.0
    
    @staticmethod
    def score_distortion(thought):
        """개별 thought 점수 (backward compatibility용)"""
        state = thought.state
        distortions = state.get('distortions', [])
        
        if not distortions:
            return 1.0
        
        valid_distortions = list(CBTScore.DISTORTION_DEFINITIONS.keys())
        
        if isinstance(distortions, str):
            import ast
            try:
                distortions = ast.literal_eval(distortions)
            except:
                distortions = [distortions]
        
        matched = any(valid_dist in str(dist).lower() for dist in distortions for valid_dist in valid_distortions)
        
        return 7.0 if matched else 3.0
    
    @staticmethod
    def score_final(thought):
        """최종 통합 점수"""
        state = thought.state
        
        has_core = 'core_beliefs' in state and state['core_beliefs']
        has_fine = 'fine_grained' in state and state['fine_grained']
        has_dist = 'distortions' in state and state['distortions']
        
        completeness = sum([has_core, has_fine, has_dist])
        
        if completeness == 0:
            return 1.0
        elif completeness == 1:
            return 3.0
        elif completeness == 2:
            return 6.0
        else:
            return 8.0
