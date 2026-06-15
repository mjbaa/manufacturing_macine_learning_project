# Manufacturing Machine Learning Project

## 1. 프로젝트 개요

이 프로젝트는 정형 제조 데이터를 기반으로 설비의 고장 여부를 예측하는 머신러닝 프로젝트입니다.

Phase 1에서는 AI4I 2020 Predictive Maintenance Dataset을 활용하여 제조 데이터에 대한 EDA를 수행하였고,
Phase 2에서는 동일한 데이터를 기반으로 정상/고장 여부를 예측하는 이진분류 모델을 구축하였습니다.

본 프로젝트의 목표는 제조 설비의 센서 데이터를 활용하여 고장 가능성을 예측하고,
고장 예측에 중요한 영향을 미치는 주요 feature를 분석하는 것입니다.

---

## 2. 사용 데이터

* Dataset: AI4I 2020 Predictive Maintenance Dataset
* Task: Binary Classification
* Target: `Machine failure`

Target 값의 의미는 다음과 같습니다.

|  값 | 의미 |
| -: | -- |
|  0 | 정상 |
|  1 | 고장 |

---

## 3. 프로젝트 구조

```text
manufacturing_macine_learning_project/
├── data/
├── notebooks/
│   └── quality_prediction_model.ipynb
├── models/
├── reports/
│   ├── feature_columns.csv
│   └── final_model_test_result.csv
├── figures/
│   ├── failure_label_distribution.png
│   ├── all_model_performance_comparison.png
│   ├── confusion_matrix_lightgbm.png
│   └── feature_importance_lightgbm.png
├── README.md
└── .gitignore
```

`data/` 폴더의 원본 CSV 파일과 `models/` 폴더의 `.pkl` 모델 파일은 `.gitignore` 설정에 따라 GitHub에는 업로드하지 않습니다.

---

## 4. 분석 및 모델링 과정

전체 모델링 과정은 다음 순서로 진행하였습니다.

1. 데이터 불러오기
2. 데이터 구조 확인
3. 컬럼명 정리
4. 이진분류 문제 정의
5. Feature / Target 분리
6. Train / Valid / Test Split
7. 평가 함수 생성
8. Logistic Regression baseline 모델 학습
9. RandomForest 모델 학습
10. XGBoost 모델 학습
11. LightGBM 모델 학습
12. 전체 모델 성능 비교
13. 최종 모델 Test 평가
14. Feature Importance 분석
15. 모델 저장
16. 최종 결론 작성

---

## 5. 전처리

모델 학습 전 다음 전처리를 수행하였습니다.

* 컬럼명을 snake_case 형태로 정리
* 식별자 컬럼 제거

  * `udi`
  * `product_id`
* 데이터 누수 방지를 위한 라벨성 컬럼 제거

  * `TWF`
  * `HDF`
  * `PWF`
  * `OSF`
  * `RNF`
* Target 컬럼 `failure` 분리
* 범주형 변수 `type`에 One-Hot Encoding 적용
* train / valid / test 데이터 분리
* 클래스 비율 유지를 위해 `stratify` 적용

`TWF`, `HDF`, `PWF`, `OSF`, `RNF`는 고장 유형을 나타내는 라벨성 컬럼이므로,
`failure` 예측 모델에 포함할 경우 데이터 누수가 발생할 수 있어 feature에서 제외하였습니다.

---

## 6. 사용 모델

본 프로젝트에서는 네 개의 모델을 학습하고 비교하였습니다.

| 모델                  | 설명                                    |
| ------------------- | ------------------------------------- |
| Logistic Regression | 이진분류 baseline 모델                      |
| RandomForest        | 여러 Decision Tree를 결합한 앙상블 모델          |
| XGBoost             | Gradient Boosting 기반 앙상블 모델           |
| LightGBM            | Gradient Boosting 기반의 빠르고 효율적인 앙상블 모델 |

불균형 데이터 문제를 고려하여 다음 설정을 적용하였습니다.

* Logistic Regression: `class_weight="balanced"`
* RandomForest: `class_weight="balanced"`
* XGBoost: `scale_pos_weight`
* LightGBM: `class_weight="balanced"`

---

## 7. 검증 데이터 성능 비교

검증 데이터 기준 모델 성능은 다음과 같습니다.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.835 |     0.137 |  0.725 |    0.230 |   0.875 |
| RandomForest        |    0.976 |     0.778 |  0.412 |    0.538 |   0.973 |
| XGBoost             |    0.967 |     0.506 |  0.765 |    0.609 |   0.977 |
| LightGBM            |    0.980 |     0.684 |  0.765 |    0.722 |   0.987 |

Logistic Regression은 Recall이 비교적 높았지만 Precision이 낮아 정상 데이터를 고장으로 잘못 예측하는 경우가 많았습니다.

RandomForest는 Precision이 높아 고장이라고 예측한 경우의 신뢰도는 높았지만, Recall이 낮아 실제 고장을 놓치는 경우가 많았습니다.

XGBoost는 Recall이 높고 F1-score도 RandomForest보다 개선되어 고장 탐지 성능이 좋아졌습니다.

LightGBM은 XGBoost와 동일한 Recall을 보이면서도 Precision이 더 높게 나타났습니다.
즉, 실제 고장을 많이 탐지하면서도 정상 데이터를 고장으로 잘못 예측하는 경우를 XGBoost보다 줄였습니다.

검증 데이터 기준으로는 F1-score와 ROC-AUC가 가장 높은 LightGBM을 최종 모델로 선정하였습니다.

---

## 8. 최종 모델 Test 평가

최종 모델로 선택한 LightGBM을 test 데이터에서 평가한 결과는 다음과 같습니다.

| Model    | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| -------- | -------: | --------: | -----: | -------: | ------: |
| LightGBM |    0.983 |     0.750 |  0.765 |    0.757 |   0.967 |

Confusion Matrix 결과는 다음과 같습니다.

| 실제 / 예측 | 정상 예측 | 고장 예측 |
| ------- | ----: | ----: |
| 실제 정상   |  1436 |    13 |
| 실제 고장   |    12 |    39 |

최종 LightGBM 모델은 실제 정상 1449개 중 1436개를 정상으로 정확히 예측하였고, 13개는 고장으로 잘못 예측하였습니다.

실제 고장 51개 중 39개를 고장으로 탐지하였고, 12개는 정상으로 잘못 예측하였습니다.

즉, 최종 모델은 실제 고장의 약 76.5%를 탐지하였으며, 고장이라고 예측한 경우 실제 고장일 비율도 75.0%로 나타났습니다.

---

## 9. 주요 시각화 결과

### Target 분포

![Failure Label Distribution](figures/failure_label_distribution.png)

### 전체 모델 성능 비교

![All Model Performance Comparison](figures/all_model_performance_comparison.png)

### 최종 모델 Confusion Matrix

![LightGBM Confusion Matrix](figures/confusion_matrix_lightgbm.png)

### Feature Importance

![LightGBM Feature Importance](figures/feature_importance_lightgbm.png)

---

## 10. Feature Importance 분석

최종 모델인 LightGBM의 Feature Importance 분석 결과는 다음과 같습니다.

| Feature        | Importance |
| -------------- | ---------: |
| `torque`       |       2328 |
| `tool_wear`    |       1994 |
| `rot_speed`    |       1655 |
| `air_temp`     |       1425 |
| `process_temp` |       1399 |
| `type_M`       |        108 |
| `type_L`       |         91 |

고장 예측에 가장 중요한 변수는 `torque`, `tool_wear`, `rot_speed`였습니다.

제조 설비 관점에서 다음과 같이 해석할 수 있습니다.

* `torque`: 설비에 걸리는 부하
* `tool_wear`: 공구의 누적 마모 정도
* `rot_speed`: 설비의 회전 속도

Phase 1 EDA에서도 고장 데이터는 정상 데이터와 비교했을 때 높은 토크, 높은 공구 마모 시간, 낮은 회전 속도와 관련되는 경향을 보였습니다.

최종 LightGBM 모델의 Feature Importance에서도 이 변수들이 상위 feature로 나타났기 때문에,
EDA에서 확인한 패턴이 모델 학습 결과와도 연결됨을 확인할 수 있었습니다.

반면 제품 유형을 나타내는 `type_M`, `type_L`의 중요도는 상대적으로 낮게 나타났습니다.
이는 제품 유형보다 센서 기반 작동 조건이 고장 예측에 더 큰 영향을 주었음을 의미합니다.

---

## 11. 모델 저장

최종 모델인 LightGBM 모델은 `models/lightgbm_binary_model.pkl` 파일로 저장하였습니다.

또한 모델 재사용을 위해 다음 파일을 저장하였습니다.

| 파일                                    | 설명                    |
| ------------------------------------- | --------------------- |
| `reports/feature_columns.csv`         | 모델 학습에 사용된 feature 목록 |
| `reports/final_model_test_result.csv` | 최종 test 성능 결과         |

모델 파일은 `.gitignore` 설정에 따라 GitHub에는 업로드하지 않았습니다.
대신 모델을 학습하고 저장하는 코드를 노트북에 포함하여 재현 가능하도록 구성하였습니다.

---

## 12. 한계점

본 프로젝트의 한계점은 다음과 같습니다.

1. 고장 데이터가 정상 데이터에 비해 적은 불균형 데이터입니다.
2. 최종 LightGBM 모델은 기존 모델보다 Precision과 Recall의 균형이 좋았지만, 여전히 실제 고장 51개 중 12개를 정상으로 예측하였습니다.
3. Feature Importance는 변수의 중요도는 보여주지만, 각 변수가 고장 확률을 높이는지 낮추는지와 같은 방향성은 설명하지 못합니다.
4. 현재 단계에서는 기본적인 모델 학습과 비교를 수행하였으며, 하이퍼파라미터 튜닝은 아직 수행하지 않았습니다.

---

## 13. 개선 방향

향후 개선 방향은 다음과 같습니다.

1. Threshold 조정을 통해 Precision과 Recall의 균형 개선
2. LightGBM 하이퍼파라미터 튜닝
3. Phase 1 EDA에서 확인한 패턴을 바탕으로 파생변수 추가

   * `torque`와 `rot_speed` 조합
   * `tool_wear`와 `torque` 조합
   * 온도 차이 feature
4. SHAP 분석을 활용한 모델 해석 강화
5. Oversampling, undersampling 등 불균형 데이터 처리 기법 적용
6. Recall 개선을 중심으로 한 모델 성능 개선

---

## 14. 실행 환경

주요 사용 라이브러리는 다음과 같습니다.

* Python
* pandas
* numpy
* matplotlib
* scikit-learn
* xgboost
* lightgbm
* joblib
* Jupyter Notebook

---

## 15. 실행 방법

1. 프로젝트를 클론합니다.

```bash
git clone https://github.com/사용자아이디/manufacturing_macine_learning_project.git
cd manufacturing_macine_learning_project
```

2. 필요한 패키지를 설치합니다.

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm joblib jupyter
```

3. `data/` 폴더에 AI4I 데이터 파일을 저장합니다.

```text
data/ai4i2020.csv
```

4. 노트북을 실행합니다.

```bash
jupyter notebook notebooks/quality_prediction_model.ipynb
```

---

## 16. 종합 결론

본 프로젝트에서는 제조 설비 데이터를 기반으로 정상/고장 여부를 예측하는 이진분류 모델을 구축하였습니다.

Logistic Regression을 baseline 모델로 사용하고, RandomForest, XGBoost, LightGBM 모델과 성능을 비교하였습니다.

검증 데이터 기준으로 LightGBM이 F1-score와 ROC-AUC에서 가장 높은 성능을 보여 최종 모델로 선택하였습니다.

최종 LightGBM 모델은 test 데이터에서 Accuracy 0.983, Precision 0.750, Recall 0.765, F1-score 0.757, ROC-AUC 0.967을 기록하였습니다.

이 모델은 정상 설비를 고장으로 잘못 판단하는 경우를 비교적 적게 유지하면서도 실제 고장의 약 76.5%를 탐지하였습니다.

Feature Importance 분석 결과, `torque`, `tool_wear`, `rot_speed`가 고장 예측에 가장 중요한 변수로 나타났습니다.

이를 통해 제조 설비의 부하, 공구 마모 정도, 회전 속도가 고장 예측에 핵심적인 역할을 한다는 것을 확인하였습니다.
