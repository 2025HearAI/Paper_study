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

여기서 가장 복잡한 항은 **\( H \times V \)**이다.  
이 복잡도를 줄이기 위해 **계층적 소프트맥스(hierarchical softmax)**를 사용하는 방법이 제안되었다.
