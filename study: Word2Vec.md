# 📄 Word2Vec 논문 리뷰

- **논문명**: *Efficient Estimation of Word Representations in Vector Space*  
- **저자**: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google)  
- **발표 연도**: 2013

## 1. Introduction

Word2Vec 논문에서는 기존의 복잡한 신경망 기반 언어 모델을 대체할 수 있는 두 가지 새로운 모델 아키텍처를 제안한다. 이들은 연산 효율성을 극대화하면서도 고품질의 단어 벡터를 학습할 수 있도록 설계되었다.

논문에서는 먼저 기존의 모델 구조인 Feedforward NNLM과 Recurrent NNLM을 설명한 뒤, 이를 바탕으로 더욱 단순하고 빠른 학습이 가능한 CBOW (Continuous Bag-of-Words)와 Skip-gram 모델을 제안한다.

## 2. Previous Work

### 2.1 Feedforward Neural Net Language Model (NNLM)
![NNLM1](https://github.com/user-attachments/assets/2de2464c-ced0-49db-914e-742cdb27d838)
![NNLM2](https://github.com/user-attachments/assets/d8fa1b36-f454-4f0c-a995-e58491b22491)

이 모델은 **input layer**, **projection layer (linear)**, **hidden layer (non-linear)**, **output layer**로 구성된다.

- **Input layer**에서는 이전 N개의 단어를 *1-of-V* 인코딩 방식으로 표현한다. (V는 어휘 크기)
- 입력층은 투영층으로 매핑되며, 이 **projection layer**는 크기가 *N × D*인 **공유된 투영 행렬(shared projection matrix)**을 사용한다.
- 단어 수 N만큼만 활성화되므로 투영층 계산은 비교적 저렴하다.

문제는 **투영층과 은닉층 사이의 연산이 밀집(dense) 행렬**이 되므로 계산이 복잡해진다는 점이다.  
**출력층**에서는 어휘 전체에 대한 확률 분포를 계산해야 하므로 출력층의 크기는 V가 된다.

따라서 학습 샘플당 계산 복잡도는 다음과 같다:

$$
Q = N \times D + N \times D \times H + H \times V
$$

여기서 가장 복잡한 항은 `H × V`이다.  
이 복잡도를 줄이기 위해 **계층적 소프트맥스(hierarchical softmax)**를 사용하는 방법이 제안되었다.

### 2.2 Recurrent Neural Net Language Model (RNNLM)
![RNNLM](https://github.com/user-attachments/assets/97cebe82-fac8-4d36-8b8a-545955b3b31c)

Feedforward NNLM은 **고정된 길이의 문맥**만 고려할 수 있다는 한계를 가진다.  
이를 극복하기 위해 **Recurrent Neural Network Language Model (RNNLM)**이 제안되었다.

RNN 모델은 **Projection Layer가 없고**, **Input Layer**, **Hidden Layer**, **Output Layer**로 구성된다.  
이 모델의 핵심은 **은닉층(Hidden Layer)**이 **자기 자신과 시간 지연된 연결을 통해 순환적으로 연결**된다는 점이다.  
즉, **이전 시간의 은닉 상태**와 **현재 입력**을 기반으로 은닉 상태를 업데이트함으로써, *short-term memory*를 형성할 수 있다.

RNN 모델의 학습 샘플당 계산 복잡도는 다음과 같다:

$$
Q = H \times H + H \times V
$$

### 2.3 Parallel Training of Neural Networks
![Parllel Training of Neural Networks](https://github.com/user-attachments/assets/9d3f014a-574d-4348-bfa7-ec283466a2fd)

대규모 데이터셋에서 신경망 모델을 학습하기 위해, Google에서는 **DistBelief**라는 대규모 분산 학습 프레임워크를 도입했다.

- 이 프레임워크는 **동일한 모델의 복제본(replica)**을 여러 개 병렬로 실행할 수 있도록 하며,  
  각 복제본은 **중앙 서버와 파라미터를 동기화**한다.
- 병렬 학습은 **미니배치 비동기적 확률적 경사하강법(mini-batch asynchronous SGD)**과  
  **적응형 학습률(Adagrad)** 알고리즘을 사용한다.

## 3. New Log-linear Models

이 절에서는 **계산 복잡도를 최소화**하면서 **분산 단어 표현(distributed word representations)**을 학습할 수 있는  
두 가지 **새로운 모델 구조**를 제안한다.

이전 장에서 살펴본 주요 관찰은, 대부분의 **계산 복잡도**가 **non-linear hidden layer**에서 발생한다는 점이다.  
이러한 비선형층은 신경망의 강점이기도 하지만, 본 논문에서는 단어를 정밀하게 표현하는 능력이 약간 떨어지더라도  
훨씬 더 **많은 데이터에 대해 효율적으로 학습**할 수 있는 **단순한 모델**을 탐색한다.

핵심 아이디어는 다음과 같다:

1. **연속적인 word vector**는 **단순한 모델**을 이용하여 학습한다.
2. 그 위에 **N-gram NNLM**은 그렇게 학습된 word vector들을 바탕으로 학습을 진행한다.
![New Log-linear Models](https://github.com/user-attachments/assets/6d9e5432-8c8c-480b-af9e-56a1a0bb09bd)
