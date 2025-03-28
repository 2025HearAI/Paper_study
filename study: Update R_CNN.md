#Vision Model
## AlexNet(2012)
ImageNet Classification with Deep Convolutional Neural Networks 
- 참고<br>
객체인식을 하기 위해서는 머신러닝을 사용한 방법들이 거의 필수적으로 다가왔고 그 성능을 끌어올리기 위해서는 많은 양의 데이터셋이나 더 강력한 모델 그리고 오버피팅을 방지하기 위한 기술들이 필요했다,
따라서 100만 개의 데이터를 학습시키기 위한 모델이 필요했고 수천 가지의 카테고리로 분류하기 위해서는 가지고 있지 않은 데이터에 대해서도 좋은 성능을 내도록 학습 시켜야 했다.
2개의 GPU로 병렬 연산을 수행하기 위해서 병렬적인 구조로 설계되었다(GTX-580 3GB*2)
- 구조<br>
![image](https://github.com/user-attachments/assets/1b72101e-6455-4caa-841a-0717684bbac8)<br>
AlexNet은 8개의 레이어로 구성되어 있다. 5개의 컨볼루션 레이어와 3개의 full-connected 레이어로 구성되어 있다. 두번째, 네번째, 다섯번째 컨볼루션 레이어들은 전 단계의 같은 채널의 특성맵들과만 연결되어 있는 반면, 세번째 컨볼루션 레이어는 전 단계의 두 채널의 특성맵들과 모두 연결되어 있다는 것을 집고 넘어가자.<br>
●각 레이어의 출력크기를 계산하는 식●<br>
![image](https://github.com/user-attachments/assets/6cba5614-158c-4a2a-88eb-e516821cf606)<br>
- 사용기법
1. Relu함수
2. GPU병렬연산
3. Overfitting 방지 기법
4. Local Response Normalization (LRN)
5. Overlapping Max Pooling

