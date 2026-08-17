# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import sys
import json
from typing import Dict, List

# 현재 디렉토리를 모듈 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from graph_of_thoughts.prompter.prompter import Prompter


class CBTPrompter(Prompter):

    extract_gap_prompt = """<Instruction>
You are a CBT therapist. Here  `{{text}}`, `{{situation}}`, `{{thoughts}}` are the user's input text, situation, and thoughts.
Extract the user's objective facts and subjective thoughts from the following inputs. The output should be in the following JSON format:
{{
    "objective_fact": "objective fact based on text and situation",
    "subjective_thought": "subjective thought based on thoughts"
}}
</Instruction>

Text: {text}
Situation: {situation}
Thoughts: {thoughts}
"""

    core_belief_prompt = """<Instruction>
[Definition of Core Beliefs]
Core beliefs are fundamental beliefs that people take for granted as absolute truths.

[Core Belief Types (3 categories)]
1. helpless
-being ineffective—in getting things done, self-protection,
and/or measuring up to other
2. unlovable
- having personal qualities resulting in an inability
to get or maintain love and intimacy from others
3. worthless
-being an immoral sinner or dangerous to others

[User Information]
Text: {text}
Situation: {situation}
Objective Fact: {of}
Subjective Thought: {st}

[Task]
Analyze the user's thoughts and identify which of the 3 core belief types are present. Select 1-3 beliefs maximum.

[Output Format - MUST USE EXACT WORDS]
{{
    "core_beliefs": ["helpless", "unlovable", "worthless"]
}}

Only include the beliefs that are clearly evident.
</Instruction>
"""

    fine_grained_core_belief_prompt = """<Instruction>
[Task]
Given the core beliefs {core_belief}, identify specific fine-grained beliefs from the user's thoughts.

[EXACT Fine-grained Categories - USE THESE EXACT PHRASES]
1. Helpless specific beliefs:
    - I am incompetent
    - I am helpless
    - I am powerless, weak, vulnerable
    - I am a victim
    - I am needy
    - I am trapped
    - I am out of control
    - I am a failure, loser
    - I am defective

2. Unlovable specific beliefs:
    - I am unlovable
    - I am unattractive
    - I am undesirable, unwanted
    - I am bound to be rejected
    - I am bound to be abandoned
    - I am bound to be alone

3. Worthless specific beliefs:
    - I am worthless, waste
    - I am immoral
    - I am bad - dangerous, toxic, evil

[User Context]
{USERINFO}

[Output Format - USE EXACT PHRASES ONLY]
{{
    "fine_grained": ["exact phrase 1", "exact phrase 2", "exact phrase 3", "exact phrase 4" .....]
}}

IMPORTANT: 
- Select 1-7 specific beliefs that match the user's thoughts
- Use ONLY the exact phrases listed above
- Match beliefs to the identified core beliefs
</Instruction>
"""

    distortion_prompt = """<Instruction>
[User Information]
Text: {text}
Situation: {situation}
Objective Fact: {of}
Subjective Thought: {st}
Core Belief: {core_belief}
Specific Belief: {specific_belief}

[Cognitive Distortions List]

* "all-or-nothing thinking": Also called black-and-white, polarized, or dichotomous thinking. You view a situation in only two categories instead of on a continuum.
* "mind reading": You believe you know what others are thinking, failing to consider other, more likely possibilities.
* "fortune-telling": Also called fortune-telling. You predict the future negatively without considering other, more likely outcomes.
* "personalization": You believe others are behaving negatively because of you, without considering more plausible explanations for their behavior.
* "emotional reasoning": You think something must be true because you "feel" (actually believe) it so strongly, ignoring or discounting evidence to the contrary.
* "overgeneralization": You make a sweeping negative conclusion that goes far beyond the current situation.
* "labeling": You put a fixed, global label on yourself or others without considering that the evidence might more reasonably lead to a less extreme conclusion.
* "should statements": Also called imperatives. You have a precise, fixed idea of how you or others should behave, and you overestimate how bad it is that these expectations are not met.
* "magnification": When you evaluate yourself, another person, or a situation, you unreasonably magnify the negative and/or minimize the positive.
* "mental filter": Also called selective abstraction. You pay undue attention to one negative detail instead of seeing the whole picture.



[Task]
Identify which cognitive distortions are present in the user's thoughts. 
Look for specific thinking patterns in their actual statements.

[Output Format - USE EXACT DISTORTION NAMES]
{{
    "distortions": ["distortion1", "distortion2", "distortion3"]
}}

IMPORTANT:
- Select multiple distortions
- Use ONLY the exact names from the list above
- Base selection on clear evidence in the user's thoughts
</Instruction>
"""

    final_integration_prompt = """<Instruction>
[Task]
Integrate all CBT analysis components into a final comprehensive analysis.

[User Information]
Text: {text}
Situation: {situation}
Objective Fact: {of}
Subjective Thought: {st}
Core Beliefs: {core_beliefs}
Fine-grained Beliefs: {fine_grained}
Cognitive Distortions: {distortions}

[Output Format]
{{
    "final_analysis": {{
        "objective_fact": "{of}",
        "subjective_thought": "{st}",
        "core_beliefs": {core_beliefs},
        "fine_grained": {fine_grained},
        "distortions": {distortions}
    }},
    "therapeutic_insights": "Brief therapeutic recommendations",
    "analysis_quality": "Assessment of analysis completeness"
}}

Provide a comprehensive final integration.
</Instruction>
"""

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        """Generate optimized prompts for CBT analysis"""
        
        # Get method from the current operation (set by CBTGenerate)
        method = getattr(self, '_current_method', None)
        
        # Fallback: try to infer method from kwargs
        if not method:
            if any(key in kwargs for key in ['core_beliefs', 'core_belief']):
                method = 'fine_grained'
            elif any(key in kwargs for key in ['fine_grained', 'specific_belief']):
                method = 'distortion'
            elif any(key in kwargs for key in ['distortions', 'distortion']):
                method = 'final_integration'
            else:
                method = 'extract_gap'  # Default fallback
        
        if method == "extract_gap":
            text = kwargs.get('ori_text', '')
            situation = kwargs.get('situation', '')
            thoughts = kwargs.get('thoughts', '')
            return self.extract_gap_prompt.format(text=text, situation=situation, thoughts=thoughts)
        
        elif method == "core_belief":
            text = kwargs.get('ori_text', '')
            situation = kwargs.get('situation', '')
            thoughts = kwargs.get('thoughts', '')
            of = kwargs.get('of', '')
            st = kwargs.get('st', '')
            return self.core_belief_prompt.format(text=text, situation=situation, of=of, st=st)
        
        elif method == "fine_grained":
            core_belief = kwargs.get('core_beliefs', [])
            userinfo = self._format_user_context(kwargs)
            return self.fine_grained_core_belief_prompt.format(
                core_belief=core_belief, 
                USERINFO=userinfo
            )
        
        elif method == "distortion":
            text = kwargs.get('ori_text', '')
            situation = kwargs.get('situation', '')
            of = kwargs.get('of', '')
            st = kwargs.get('st', '')
            core_belief = kwargs.get('core_beliefs', '')
            specific_belief = kwargs.get('fine_grained', '')
            
            return self.distortion_prompt.format(
                text=text, situation=situation, of=of, st=st,
                core_belief=core_belief, specific_belief=specific_belief
            )
        
        elif method == "final_integration":
            text = kwargs.get('ori_text', '')
            situation = kwargs.get('situation', '')
            of = kwargs.get('of', '')
            st = kwargs.get('st', '')
            core_beliefs = kwargs.get('core_beliefs', [])
            fine_grained = kwargs.get('fine_grained', [])
            distortions = kwargs.get('distortions', [])
            
            return self.final_integration_prompt.format(
                text=text, situation=situation, of=of, st=st,
                core_beliefs=core_beliefs, fine_grained=fine_grained, distortions=distortions
            )
        
        else:
            # Fallback to extract_gap for unknown methods
            text = kwargs.get('ori_text', '')
            situation = kwargs.get('situation', '')
            thoughts = kwargs.get('thoughts', '')
            return self.extract_gap_prompt.format(text=text, situation=situation, thoughts=thoughts)

    def _format_user_context(self, kwargs):
        """Format user context for prompts"""
        text = kwargs.get('ori_text', '')
        situation = kwargs.get('situation', '')
        thoughts = kwargs.get('thoughts', '')
        
        return f"Text: {text}\nSituation: {situation}\nThoughts: {thoughts}"

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """Generate aggregation prompt"""
        return "Aggregate the provided analysis results."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """Generate scoring prompt using CBTScore class"""
        if not state_dicts:
            return "No analysis to score."
        
        # Use existing CBTScore class for proper LLM scoring prompts
        from cbt_score import CBTScore
        
        method = getattr(self, '_current_method', None)
        
        # Convert method to category for CBTScore
        if method == 'core_belief_score':
            category = 'core_beliefs'
        elif method == 'fine_grained_score':
            category = 'fine_grained'
        elif method == 'distortion_score':
            category = 'distortions'
        elif method == 'final':
            # For final scoring, check completeness
            state = state_dicts[0] if state_dicts else {}
            has_core = bool(state.get('core_beliefs'))
            has_fine = bool(state.get('fine_grained'))
            has_dist = bool(state.get('distortions'))
            
            completeness = sum([has_core, has_fine, has_dist])
            
            return f"""Rate the completeness of this CBT analysis on a scale of 1-10:

Analysis Components:
- Core beliefs: {state.get('core_beliefs', 'Missing')}
- Fine-grained beliefs: {state.get('fine_grained', 'Missing')} 
- Cognitive distortions: {state.get('distortions', 'Missing')}

Completeness: {completeness}/3 components present

Only provide the score as a number (e.g., 8):"""
        else:
            return "Rate the analysis quality on a scale of 1-10. Only provide the score as a number (e.g., 7):"
        
        # Create mock thoughts for frequency analysis
        mock_thoughts = []
        for state in state_dicts:
            mock_thought = type('MockThought', (), {'state': state})()
            mock_thoughts.append(mock_thought)
        
        # Use CBTScore for frequency analysis and LLM prompt generation
        analysis_result = CBTScore.analyze_frequency(mock_thoughts, category)
        
        # Generate context from kwargs
        context = {
            'text': kwargs.get('ori_text', ''),
            'situation': kwargs.get('situation', ''),
            'thoughts': kwargs.get('thoughts', ''),
            'core_beliefs': kwargs.get('core_beliefs', ''),
            'fine_grained': kwargs.get('fine_grained', ''),
            'distortions': kwargs.get('distortions', '')
        }
        
        # Generate LLM scoring prompt
        return CBTScore.generate_llm_scoring_prompt(category, analysis_result, context)

    def improve_prompt(self, **kwargs) -> str:
        """Generate improvement prompt"""
        return "Improve the analysis results."

    def validation_prompt(self, **kwargs) -> str:
        """Generate validation prompt"""
        return "Validate the analysis results."

    def userInput_prompt(self, **kwargs) -> str:
        """Generate user input prompt"""
        return "Process user input."


if __name__ == "__main__":
    print("CBTPrompter module - use via cbt_main.py")