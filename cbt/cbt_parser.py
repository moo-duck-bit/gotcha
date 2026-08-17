# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import sys
import json
import re
from typing import Dict, List, Union

# 현재 디렉토리를 모듈 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from graph_of_thoughts.parser.parser import Parser
from graph_of_thoughts.language_models import AbstractLanguageModel
from cbt_utils import parse_list_string
from cbt_score import CBTScore

# Mock Thought class for scoring compatibility
class MockThought:
    def __init__(self, state):
        self.state = state

class CBTParser(Parser):
    """Optimized CBT analysis response parser"""

    def __init__(self) -> None:
        """Initialize parser"""
        self.json_pattern = re.compile(r'{.*}', re.DOTALL)
        self.scorer = CBTScore()

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON data from text with improved accuracy"""
        import re
        
        # Try multiple JSON extraction patterns
        patterns = [
            r'\{[^{}]*"[^"]*":\s*\[[^\]]*\][^{}]*\}',  # JSON with arrays
            r'\{[^{}]*"[^"]*":\s*"[^"]*"[^{}]*\}',     # JSON with strings
            r'\{.*?\}',                                 # General JSON
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if result:  # Non-empty result
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Fallback: manual extraction
        result = {}
        
        # Extract core_beliefs
        core_match = re.search(r'"?core_beliefs"?\s*:\s*\[(.*?)\]', text, re.IGNORECASE | re.DOTALL)
        if core_match:
            beliefs_str = core_match.group(1)
            beliefs = [b.strip(' "\'') for b in beliefs_str.split(',') if b.strip()]
            result['core_beliefs'] = beliefs
        
        # Extract fine_grained
        fine_match = re.search(r'"?fine_grained"?\s*:\s*\[(.*?)\]', text, re.IGNORECASE | re.DOTALL)
        if fine_match:
            fine_str = fine_match.group(1)
            fine_beliefs = [b.strip(' "\'') for b in fine_str.split(',') if b.strip()]
            result['fine_grained'] = fine_beliefs
        
        # Extract distortions
        dist_match = re.search(r'"?distortions"?\s*:\s*\[(.*?)\]', text, re.IGNORECASE | re.DOTALL)
        if dist_match:
            dist_str = dist_match.group(1)
            distortions = [d.strip(' "\'') for d in dist_str.split(',') if d.strip()]
            result['distortions'] = distortions
        
        return result

    def _extract_core(self, text: str) -> str:
        """Extract core beliefs with improved accuracy"""
        json_data = self._extract_json(text)
        if 'core_beliefs' in json_data:
            beliefs = json_data['core_beliefs']
            if isinstance(beliefs, list):
                # Clean and validate beliefs
                valid_beliefs = []
                valid_types = ['helpless', 'unlovable', 'worthless', 'untrustworthy', 'rejecting', 'dangerous', 'unfair']
                
                for belief in beliefs:
                    belief_clean = str(belief).lower().strip()
                    for valid_type in valid_types:
                        if valid_type in belief_clean:
                            valid_beliefs.append(valid_type)
                            break
                
                return str(valid_beliefs) if valid_beliefs else str(beliefs)
        
        # Fallback pattern matching with better accuracy
        valid_types = ['helpless', 'unlovable', 'worthless', 'untrustworthy', 'rejecting', 'dangerous', 'unfair']
        found_beliefs = []
        
        for belief_type in valid_types:
            if belief_type in text.lower():
                found_beliefs.append(belief_type)
        
        if found_beliefs:
            return str(found_beliefs)
            
        return "Core belief extraction failed"

    def _extract_fine_grained(self, text: str) -> str:
        """Extract fine-grained beliefs with improved accuracy"""
        json_data = self._extract_json(text)
        if 'fine_grained' in json_data:
            beliefs = json_data['fine_grained']
            if isinstance(beliefs, list):
                # Clean fine-grained beliefs
                cleaned_beliefs = []
                for belief in beliefs:
                    belief_str = str(belief).strip(' "\'')
                    if len(belief_str) > 3:  # Valid belief should have some content
                        cleaned_beliefs.append(belief_str)
                
                return str(cleaned_beliefs) if cleaned_beliefs else str(beliefs)
        
        # Fallback: extract common fine-grained patterns
        fine_patterns = [
            r'I am (helpless|incompetent|powerless|weak|vulnerable|defective)',
            r'I am (unlovable|undesirable|unwanted|bound to be rejected|bound to be abandoned|bound to be alone)',
            r'I am (worthless|waste|immoral|bad|a failure|loser)'
        ]
        
        found_beliefs = []
        for pattern in fine_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    found_beliefs.extend([f"I am {m}" for m in match if m])
                else:
                    found_beliefs.append(f"I am {match}")
        
        if found_beliefs:
            return str(found_beliefs)
            
        return "Fine-grained beliefs extraction failed"

    def _extract_distortions(self, text: str) -> str:
        """Extract cognitive distortions with improved accuracy"""
        json_data = self._extract_json(text)
        if 'distortions' in json_data:
            distortions = json_data['distortions']
            if isinstance(distortions, list):
                # Validate distortions against known types
                valid_distortions = []
                known_distortions = [
                    'all-or-nothing thinking', 'mind reading', 'fortune-telling', 
                    'personalization', 'emotional reasoning', 'overgeneralization',
                    'labeling', 'should statements', 'magnification', 'mental filter'
                ]
                
                for distortion in distortions:
                    distortion_clean = str(distortion).lower().strip()
                    for known in known_distortions:
                        if known in distortion_clean or distortion_clean in known:
                            valid_distortions.append(known)
                            break
                
                return str(valid_distortions) if valid_distortions else str(distortions)
        
        # Fallback: pattern matching for distortions
        known_distortions = [
            'all-or-nothing thinking', 'mind reading', 'fortune-telling', 
            'personalization', 'emotional reasoning', 'overgeneralization',
            'labeling', 'should statements', 'magnification', 'mental filter'
        ]
        
        found_distortions = []
        for distortion in known_distortions:
            variations = [
                distortion,
                distortion.replace('-', ' '),
                distortion.replace('-', ''),
                distortion.replace(' thinking', ''),
                distortion.replace(' reasoning', '')
            ]
            
            for variation in variations:
                if variation in text.lower():
                    found_distortions.append(distortion)
                    break
        
        if found_distortions:
            return str(found_distortions)
            
        return "Cognitive distortions extraction failed"

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """Parse Generate operation responses"""
        method = state.get('method', '')
        print(f"[{method}] Processing {len(texts)} responses")

        if not texts:
            print(f"[{method}] Warning: No responses received")
            return [{}]
            
        results = []
        for idx, text in enumerate(texts):
            result = self._extract_json(text)
            changes = {}
            
            if method == 'extract_gap':
                if 'objective_fact' in result and 'subjective_thought' in result:
                    changes['of'] = result['objective_fact']
                    changes['st'] = result['subjective_thought']
                else:
                    # Fallback extraction
                    of_match = re.search(r'objective.*?[:"](.*?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
                    st_match = re.search(r'subjective.*?[:"](.*?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
                    
                    changes['of'] = of_match.group(1).strip() if of_match else "Extraction failed"
                    changes['st'] = st_match.group(1).strip() if st_match else "Extraction failed"
                    
            elif method == 'core_belief':
                changes['core_beliefs'] = self._extract_core(text)
                
            elif method == 'fine_grained':
                changes['fine_grained'] = self._extract_fine_grained(text)
                
            elif method == 'distortion':
                changes['distortions'] = self._extract_distortions(text)
                
            elif method == 'final_integration':
                # Extract all components for final integration
                final_result = self._extract_json(text)
                if final_result:
                    changes.update(final_result)
                
            results.append(changes)
        
        return results

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """Parse scoring responses using simple LLM scores"""
        scores = []
        
        for i, (state, text) in enumerate(zip(states, texts)):
            method = state.get('method', '')
            
            # LLM 응답에서 숫자 점수 추출
            llm_score = self._extract_score_from_text(text)
            if llm_score is not None:
                scores.append(llm_score)
                print(f"[{method}] LLM Score: {llm_score:.1f}")
                continue
            
            # Fallback scoring
            mock_thought = MockThought(state)
            
            # Use CBTScore methods
            if method == 'core_belief_score':
                score = self.scorer.score_core_belief(mock_thought)
            elif method == 'fine_grained_score':
                score = self.scorer.score_fine_grained(mock_thought)
            elif method == 'distortion_score':
                score = self.scorer.score_distortion(mock_thought)
            elif method == 'final':
                score = self.scorer.score_final(mock_thought)
            else:
                score = 1.0
            
            scores.append(score)
            print(f"[{method}] Candidate {i+1} Score: {score:.1f}")
        
        return scores

    def _extract_score_from_text(self, text: str) -> float:
        """Extract numeric score from LLM response text"""
        import re
        
        # 숫자 패턴 찾기 (1-10 범위)
        number_patterns = [
            r'\b([1-9]|10)(?:\.\d+)?\b',  # 1-10 범위의 숫자
            r'Score:\s*([1-9]|10)(?:\.\d+)?',  # "Score: 8" 형태
            r'(\d+)(?:\.\d+)?(?:\s*/\s*10)?',  # 일반 숫자 패턴
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    score = float(matches[0])
                    if 1.0 <= score <= 10.0:
                        return score
                except (ValueError, IndexError):
                    continue
        
        return None

    def _check_definition_match(self, category: str, values: List[str]) -> bool:
        """정의와의 매칭 확인"""
        if category == 'core_beliefs':
            valid_cores = ['helpless', 'unlovable', 'worthless']
            return any(valid in str(val).lower() for val in values for valid in valid_cores)
        
        elif category == 'fine_grained':
            all_defined = []
            for beliefs in self.scorer.FINE_GRAINED_DEFINITIONS.values():
                all_defined.extend(beliefs)
            return any(defined.lower() in str(val).lower() for val in values for defined in all_defined)
        
        elif category == 'distortions':
            valid_dists = list(self.scorer.DISTORTION_DEFINITIONS.keys())
            return any(valid in str(val).lower() for val in values for valid in valid_dists)
        
        return False

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """Parse aggregation responses"""
        if len(texts) == 1:
            return self._extract_json(texts[0])
        return [self._extract_json(text) for text in texts]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """Parse improvement responses"""
        return self._extract_json(texts[0]) if texts else {}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """Parse validation responses"""
        return "valid" in texts[0].lower() if texts else False 