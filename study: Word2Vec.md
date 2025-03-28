# 📄 Word2Vec 논문 리뷰

- **논문명**: *Efficient Estimation of Word Representations in Vector Space*  
- **저자**: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google)  
- **발표 연도**: 2013

## 1. Introduction

Word2Vec 논문에서는 기존의 복잡한 신경망 기반 언어 모델을 대체할 수 있는 두 가지 새로운 모델 아키텍처를 제안한다. 이들은 연산 효율성을 극대화하면서도 고품질의 단어 벡터를 학습할 수 있도록 설계되었다.

논문에서는 먼저 기존의 모델 구조인 Feedforward NNLM과 Recurrent NNLM을 설명한 뒤, 이를 바탕으로 더욱 단순하고 빠른 학습이 가능한 CBOW (Continuous Bag-of-Words)와 Skip-gram 모델을 제안한다.
