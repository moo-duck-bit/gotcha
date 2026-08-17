# CBT Dataset Evaluation System

🧠 **Graph of Thoughts 기반 CBT 인지편향 분석 평가 시스템**

이 시스템은 Beck의 인지 모델과 Graph of Thoughts 프레임워크를 활용하여 CBT(인지행동치료) 관련 데이터셋을 평가합니다.

## 📊 지원 데이터셋

| 데이터셋 | 파일 | 대상 | 설명 |
|---------|------|------|------|
| **핵심신념** | `CBT-Bench/core/core_major_test.json` | `core_belief_major` | Beck의 3대 핵심신념 (helpless, unlovable, worthless) |
| **세부신념** | `CBT-Bench/fine/core_fine_test.json` | `fine_grained` | 구체적이고 세분화된 인지적 신념 패턴 |
| **인지왜곡** | `CBT-Bench/distortions/distortions_test.json` | `distortions` | Burns의 10가지 인지왜곡 유형 |

## 🎯 평가 지표

시스템은 다음 4가지 지표로 성능을 평가합니다:

- **Precision (정밀도)**: 예측한 항목 중 실제 정답인 비율
- **Recall (재현율)**: 실제 정답 중 올바르게 예측한 비율  
- **F1-Score**: Precision과 Recall의 조화평균
- **Accuracy (정확도)**: 완전히 일치하는 예측의 비율

## 🚀 사용법

### 1. 전체 데이터셋 평가

```bash
# 모든 데이터셋 평가 (기본)
./evaluate_datasets.sh

# 모든 데이터셋을 각각 10개 케이스만 평가
./evaluate_datasets.sh --num-cases 10

# 특정 설정 파일 사용
./evaluate_datasets.sh --config my_config.json
```

### 2. 개별 데이터셋 평가

```bash
# 핵심신념만 평가
./evaluate_core_beliefs.sh

# 세부신념만 평가  
./evaluate_fine_grained.sh

# 인지왜곡만 평가
./evaluate_distortions.sh

# 개별 데이터셋에 케이스 수 제한
./evaluate_core_beliefs.sh --num-cases 5
```

### 3. 고급 옵션

```bash
# 특정 데이터셋만 평가
./evaluate_datasets.sh --dataset core_major

# 결과 저장 없이 실행 (빠른 테스트용)
./evaluate_datasets.sh --no-save-details

# 다양한 옵션 조합
./evaluate_datasets.sh --dataset distortions --num-cases 20 --config custom.json
```

## 📋 명령어 옵션

### evaluate_datasets.sh 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-d, --dataset` | 평가할 데이터셋 (core_major\|core_fine\|distortions\|all) | all |
| `-n, --num-cases` | 평가할 케이스 수 | 모든 케이스 |
| `-c, --config` | 언어모델 설정 파일 경로 | config.json |
| `--no-save-details` | 상세 결과 JSON 저장 안 함 | 저장함 |
| `-h, --help` | 도움말 출력 | - |

## 📁 결과 파일

평가 완료 후 `cbt/results/` 디렉토리에 다음 파일들이 생성됩니다:

```
cbt/results/
├── core_major_evaluation_20241201_143022.json      # 핵심신념 상세 결과
├── core_fine_evaluation_20241201_143045.json       # 세부신념 상세 결과  
├── distortions_evaluation_20241201_143108.json     # 인지왜곡 상세 결과
└── combined_evaluation_20241201_143108.json        # 통합 결과
```

### 결과 파일 구조

```json
{
  "dataset_info": {
    "name": "core_major",
    "display_name": "핵심신념 (Core Beliefs)",
    "total_cases": 0,
    "successful_cases": 0,
    "success_rate": 0.0
  },
  "performance_metrics": {
    "precision": {"mean": 0.0, "std": 0.0},
    "recall": {"mean": 0.0, "std": 0.0},
    "f1": {"mean": 0.0, "std": 0.0},
    "accuracy": {"mean": 0.0, "std": 0.0}
  },
  "detailed_results": [...]
}
```

## 📊 출력 예시

실행 시 다음과 같은 CBT-CD 형태의 테이블이 출력됩니다:

```
================================================================================
                            CBT-CD Evaluation Results                            
================================================================================
Dataset              Precision    Recall       F1           Accuracy    
--------------------------------------------------------------------------------
핵심신념 (Core Be      --.-         --.-         --.-         --.-        
세부신념 (Fine-gr      --.-         --.-         --.-         --.-        
인지왜곡 (Cogniti      --.-         --.-         --.-         --.-        
--------------------------------------------------------------------------------
AVERAGE              --.-         --.-         --.-         --.-        
================================================================================
```

## 🔧 문제 해결

### 일반적인 오류

1. **데이터셋 파일 없음**
   ```bash
   ❌ Core beliefs dataset not found: CBT-Bench/core/core_major_test.json
   ```
   → 데이터셋 파일 경로를 확인하세요.

2. **Graph of Thoughts 라이브러리 없음**
   ```
   Graph of Thoughts not available: No module named 'graph_of_thoughts'
   ```
   → Graph of Thoughts 라이브러리를 설치하세요.

3. **설정 파일 없음**
   ```bash
   ❌ Config file not found: config.json
   ```
   → `config.json` 파일을 생성하거나 `--config` 옵션으로 경로를 지정하세요.

### 권한 오류

스크립트 실행 권한이 없는 경우:
```bash
chmod +x evaluate_datasets.sh evaluate_core_beliefs.sh evaluate_fine_grained.sh evaluate_distortions.sh
```

## 🧪 빠른 테스트

시스템을 빠르게 테스트하려면:

```bash
# 각 데이터셋당 3개 케이스만 빠르게 테스트
./evaluate_datasets.sh --num-cases 3

# 핵심신념만 5개 케이스로 테스트
./evaluate_core_beliefs.sh --num-cases 5
```

## 📚 이론적 배경

### Beck's Cognitive Model (1979)
- **핵심신념**: 자신, 세계, 미래에 대한 근본적 믿음
- **인지적 삼각형**: 생각-감정-행동의 상호작용
- **3대 핵심신념**: Helpless, Unlovable, Worthless

### Burns' Cognitive Distortions
1. All-or-Nothing Thinking
2. Overgeneralization  
3. Mental Filter
4. Disqualifying the Positive
5. Jumping to Conclusions
6. Magnification/Minimization
7. Emotional Reasoning
8. Should Statements
9. Labeling
10. Personalization

### Graph of Thoughts Framework (Besta et al., 2024)
- **Sequential Processing**: 단계별 인지 분석
- **Parallel Evaluation**: 다중 후보 동시 평가
- **Best Selection**: 최적 결과 선택 메커니즘

## 📞 지원

문제가 발생하거나 추가 기능이 필요한 경우, 다음을 확인하세요:

1. 모든 의존성이 설치되었는지 확인
2. 데이터셋 파일이 올바른 위치에 있는지 확인
3. 설정 파일이 올바르게 구성되었는지 확인
4. 로그 파일과 오류 메시지 검토

---

🧠 **Powered by Graph of Thoughts** | 📚 **Based on Beck's Cognitive Model** | 🎯 **CBT Analysis at Scale** 