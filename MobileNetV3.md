# 📄 MobileNetV3 논문 리뷰

- **논문명**: Searching for MobileNetV3  
- **저자**: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam (Google AI / Google Brain)  
- **발표 연도**: 2019  

---

## 1. Abstract

MobileNetV3는 하드웨어 인지형 신경망 아키텍처 검색(Network Architecture Search, NAS)과 기존의 NetAdapt 알고리즘을 결합하여 설계된 고효율 모바일 신경망 모델이다. 이 논문에서는 고성능 모바일 기기에서 사용할 수 있는 두 가지 모델(MobileNetV3-Large, MobileNetV3-Small)을 제안하며, 분류, 객체 탐지, 의미 분할 등 다양한 컴퓨터 비전 과제에서 기존 모델 대비 향상된 정확도 및 처리 속도를 보여준다.

## 2. Efficient Mobile Building Blocks

MobileNet 시리즈는 모바일 환경에서의 효율적인 연산을 목표로 지속적으로 개선되어 왔다. MobileNetV3는 이러한 발전의 흐름 속에서 등장한 모델로, 이전 세대의 구조적 장점을 계승하고 새로운 기법을 결합하여 더욱 효율적인 블록 구조를 완성했다. 이 절에서는 MobileNetV1부터 V3까지의 핵심 빌딩 블록과 그 발전 과정을 정리하고, MobileNetV3에서 최종적으로 사용된 구성 요소들을 설명한다.

---

### 2.1 MobileNetV1

MobileNetV1은 경량 신경망 구조를 위해 depthwise separable convolution을 도입하였다. 이는 전통적인 convolution을 두 단계로 분해하여 계산량을 획기적으로 줄이는 방식이다.

- **구성**: Depthwise convolution → Pointwise (1×1) convolution  
- **장점**: 연산량 및 파라미터 수 감소 (예: 기존 Conv 대비 약 8~9배 적은 연산량)  
- **한계**: 구조는 단순하지만 표현력이 제한적이며, 이후 세대에서 보완됨

---

### 2.2 MobileNetV2
![mobilenetv2](https://github.com/user-attachments/assets/8d8d2352-1ae8-434c-8b67-8f3619574856)

MobileNetV2는 V1의 구조를 기반으로 inverted residual block과 linear bottleneck을 도입해 연산 효율성과 표현력을 동시에 개선하였다.

- **핵심 구조**:
  - Expansion: 채널 수를 확장 (1×1 Conv)
  - Depthwise convolution: 공간 필터링
  - Projection: 채널 수 축소 (1×1 Conv)
  - Residual connection: 입력과 출력이 같은 경우 연결

- **특징**:
  - 비선형 활성 함수는 bottleneck 영역에는 사용하지 않음  
  - 저차원 → 고차원 → 저차원으로 정보 흐름 구성  
  - 연산 효율성과 정확도의 균형 유지

---

### 2.3 MnasNet

MnasNet은 **플랫폼 인식형 NAS(Neural Architecture Search)**를 적용해 자동으로 효율적인 아키텍처를 탐색한 모델이다.

- **핵심 기법**: Reinforcement Learning을 기반으로 한 RNN Controller 사용  
- **구성 요소**:
  - MobileNetV2 구조 기반
  - Squeeze-and-Excitation(SE) 모듈 추가 → 채널 간 상호작용 강화

- **특징**:
  - 정확도와 지연 시간을 동시에 고려한 보상 함수 설계  
  - 실제 하드웨어(Pixel 1)의 성능을 기준으로 최적화  

---

### 2.4 MobileNetV3
![mobilenetv3](https://github.com/user-attachments/assets/12fdf005-d858-4b48-9037-35893050dd53)

MobileNetV3는 앞선 세대들의 장점을 종합하면서 다음과 같은 개선 사항을 적용했다.

- **하이브리드 구조**: MobileNetV2 블록 + MnasNet의 SE 모듈 + 새로운 비선형 함수  
- **활성 함수 개선**:
  - Swish 대신 계산량이 적은 Hard-Swish(h-swish) 사용  
  - Sigmoid → Hard-Sigmoid로 대체하여 정수 연산 친화적으로 개선

- **SE 모듈 위치 변경**: ResNet 스타일 대신, expansion 영역 이후에 배치하여 효율 증가  

- **최종 구조**:
  - MobileNetV3-Large: 고성능 모바일 환경  
  - MobileNetV3-Small: 경량 디바이스 및 IoT 환경

## 3. Network Search

MobileNetV3는 네트워크 구조의 설계를 자동화하기 위해 두 가지 아키텍처 탐색 기법을 활용한다. 하나는 전체 블록 수준에서의 Platform-Aware NAS이고, 다른 하나는 레이어 수준의 세밀한 조정을 위한 NetAdapt 알고리즘이다. 이 두 기법은 상호 보완적인 특성을 가지며, 함께 사용됨으로써 모바일 하드웨어에서의 정확도–지연 시간 트레이드오프를 최적화할 수 있다.

---

### 3.1 Platform-Aware NAS for Block-wise Search

Platform-Aware NAS는 블록 단위로 네트워크의 전반적인 구조를 탐색하는 방식으로, 실제 모바일 기기에서의 연산 속도를 고려하여 설계된다.

- **탐색 방식**:
  - RNN 기반 컨트롤러 사용  
  - MnasNet-A1 구조를 초기 시드 모델로 채택  

- **보상 함수**:  
  - 정확도와 지연 시간을 모두 고려한 다목적 함수  
    - ![3 1-보상함수](https://github.com/user-attachments/assets/16954b3a-8a62-4b0a-a985-d5a914898cc2)
    (단, \( w \)는 모델 규모에 따라 조정됨)

- **Large vs. Small 모델 차이**:  
  - Small 모델은 latency에 따라 정확도 변화폭이 크므로 weight 조정 필요  
    → \( w = -0.15 \) (Large 모델에선 \( w = -0.07 \))

이 기법을 통해 MobileNetV3의 초기 구조가 결정되며, 이후 세부 튜닝을 위해 NetAdapt이 적용된다.

---

### 3.2 NetAdapt for Layer-wise Search

NetAdapt은 레이어별 필터 수를 미세 조정하여 전체 네트워크 구조를 더욱 정밀하게 최적화하는 알고리즘이다.

- **절차**:
  1. NAS로 생성된 시드 모델에서 시작  
  2. 각 단계에서 latency를 일정 비율 줄이는 여러 후보 아키텍처 제안  
  3. 기존 모델의 학습된 가중치를 재사용하며 빠르게 fine-tuning  
  4. 정확도/지연 시간 비율이 가장 높은 후보 선택  
  5. 목표 latency 도달 시까지 반복  

- **선택 기준**:  
  - ![3-2선택기준](https://github.com/user-attachments/assets/d93e1304-5695-46bd-bbd0-f44211f6bef8)

- **하이퍼파라미터**:
  - Fine-tuning step: 10,000  
  - 최소 latency 감소량 ![최소 latency 감소량](https://github.com/user-attachments/assets/aa826b08-e4a5-4f83-8097-ec86a67181cf)

이 기법은 각 레이어의 중요도와 성능 민감도를 고려해 모델의 연산 효율성을 극대화하는 데 기여한다.

## 4. Network Improvement

MobileNetV3는 자동 아키텍처 탐색 결과를 그대로 사용하는 것이 아니라, 수동적인 구조 개선과 연산 최적화를 통해 성능과 효율성을 더욱 향상시켰다. 이 섹션에서는 주요한 네트워크 개선 요소들을 다음과 같이 정리한다:

---

### 4.1 초반 및 후반 레이어의 구조 재설계

모바일 환경에서 가장 연산량이 많은 레이어는 모델의 시작과 끝부분이다. MobileNetV3는 이 구간을 다음과 같이 최적화하였다.

- **마지막 레이어**
  - 기존: 7×7 해상도에서 1×1 Conv 수행 → 연산량 높음
  - 개선: Average Pooling 이후 1×1 Conv 수행 → 1×1 해상도로 줄여 연산량 감소
  - **효과**: 약 11% latency 감소, 3개의 연산 블록 제거 (약 30M MAdds 절감)

- **초기 Conv 필터 수 축소**
  - 기존: 32 filters 사용
  - 개선: 16 filters + h-swish 적용
  - **효과**: 정확도 유지하며 latency 2ms 감소

---

### 4.2 활성화 함수 최적화 (Nonlinearity)

Swish 함수는 정확도는 높지만 sigmoid 연산 비용이 크다. 이를 해결하기 위해 MobileNetV3에서는 **Hard-Swish (h-swish)** 함수가 사용되었다.

- **Hard-Swish 정의**:
  ![h-swish (Hard-Swish)](https://github.com/user-attachments/assets/efadab8b-3337-483f-863a-fc8fe6eb1048)

- **장점**
  - 정수 연산 및 양자화에 유리
  - 최적화된 연산 구현 가능 (piece-wise 함수)
  - 정확도 손실 없음

- **사용 전략**
  - 네트워크 초반부: ReLU 사용
  - 네트워크 후반부: h-swish 사용
  - **효과**: Pixel 기기 기준, 최적 구현 시 약 6ms latency 절감

---

### 4.3 Squeeze-and-Excitation (SE) 모듈 개선

SE 모듈은 채널 간 중요도를 학습하는 구조로, MobileNetV3에서는 다음과 같이 개선되었다.

- 기존: Expansion layer 채널 수의 1/6 또는 1/16 사용
- 개선: **1/4 비율 고정** → 약간의 파라미터 증가로 정확도 향상, latency 변화 없음
- **위치 변경**: ResNet 스타일과 달리, **depthwise conv 이후**에 배치 → 더 큰 표현 공간에서 attention 수행

---

### 4.4 최종 MobileNetV3 구조

이러한 개선을 통해 최종적으로 아래 두 가지 모델이 완성되었다:

- **MobileNetV3-Large**: 고성능 환경에 최적화
- **MobileNetV3-Small**: 저전력, 초경량 환경에 최적화

> 두 모델 모두 NAS + NetAdapt + 구조 최적화 + 새로운 활성 함수가 결합된 형태로 설계됨

## 5. Summary

MobileNetV3는 모바일 기기에서의 효율적이고 정확한 인공지능 처리를 위해 고안된 경량 신경망 모델이다.  
이 모델은 자동 아키텍처 탐색(NAS), 세밀한 구조 조정(NetAdapt), 그리고 수작업 최적화를 결합하여 설계되었으며,  
MobileNetV1과 V2, MnasNet의 핵심 아이디어를 계승 및 발전시켰다.

### MobileNetV3의 주요 특징은 다음과 같다:

- **효율적 구조 구성**  
  V1의 Depthwise Separable Conv, V2의 Inverted Residual Block, MnasNet의 SE 모듈을 통합하고 개선함

- **자동 설계 + 수작업 개선**  
  NAS로 전반적인 블록 구조를 탐색하고, NetAdapt로 레이어 단위 조정 후, 수동 최적화로 효율성을 극대화

- **새로운 활성화 함수**  
  Swish의 계산량 문제를 해결하기 위해 하드 버전인 `h-swish` 도입 (양자화 친화적, 속도 향상)

- **구조 최적화**  
  연산량이 큰 초기 및 말단 레이어를 재구성하여 latency를 줄이면서도 정확도 유지

- **모델 이원화**  
  V3-Large는 고성능 기기용, V3-Small은 저전력 환경용으로 각각 최적화됨

---

결과적으로 MobileNetV3는 분류, 객체 탐지, 의미 분할 등 다양한 컴퓨터 비전 과제에서  
기존 모델보다 우수한 **성능–효율 트레이드오프**를 달성하며,  
모바일 AI 모델 설계의 **새로운 기준**을 제시한다.
