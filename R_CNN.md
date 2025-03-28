## R-CNN 논문 리뷰

**Object Detection**
> Classification: 한 물체가 아닌 여러 물체에 대해 어떤 물체인지 클래스 분류 <br>
> Localization: 물체가 어디 있는지 박스를 통해 위치 정보를 나타냄

### (Multi-Labeled) Classification + Localization
![image](https://github.com/user-attachments/assets/0dcea5ef-febe-4fe3-9630-bcef08368d6b)

- 1-stage Detector : Classification, Localization을 동시에 
- 2-stage Detector : Classification, Localization을 순차적으로
#### 1-stage가 비교적 빠르지만 정확도 낮음
#### 2-stage가 비교적 느리지만 정확도 높음 
<br>

---
## R-CNN (Regions with Convolutional Neural Networks features)
### 설정한 Region을 CNN의 Feature(입력값)로 활용하여 Object Detection을 수행하는 신경망


### 모델 구조
![image](https://github.com/user-attachments/assets/32f65056-a06f-4ebe-bd91-3edbc3699eec)


**1. Image 입력** <br>
**2. Region proposal: 물체 위치 찾음** <br>
**2. Classification & Regression: 물체 분류**  <br>
 <br>

1. 이미지에 있는 데이터와 레이블을 투입한 후 카테고리에 무관하게 물체의 영역을 찾는 Region Proposal.
2. proposal 된 영역으로부터 고정된 크기의 Feature Vector를 warping/crop 하여 CNN의 인풋으로 사용. <br> CNN은 ImageNet 사전훈련된 네트워크 사용
3. CNN을 통해 나온 feature map을 활용하여 선형 지도학습 모델인 SVM(Support Vector Machine)을 통한 분류
4. Regressor를 통한 bounding box regression 진행

<br>

**Region Proposal**
- 데이터와 레이블 투입
- R-CNN에는 일반적인 이미지를 데이터로 사용, 레이블은 정답 Bounding Box를 줌
- 이미지를 vertex로 표현하고, 그 vertex들의 연결을 edge로 보면 됨
> G=(V,E)로 표현되는 vertex(node)와 Edge

- R-CNN에서는 Selective search이라는 알고리즘을 활용하여 객체가 있을 수 있는 영역 제안
 ![image](https://github.com/user-attachments/assets/1650be00-3560-4405-a3af-3f86a3aba9a7)
- **Selective Search**는 위 그림과 같이 이미지를 수많은 작은 영역으로 분할한 뒤 명암 차이 등의 그룹화 기준을 가지고 영역들을 합치는 **Bottom-up방식의 region proposal** 방법

![image](https://github.com/user-attachments/assets/6d941838-0792-455d-947f-1b514c5f447c)

Region Proposal 단계에서 물체가 있을 법한 영역을 찾음
기존의 Sliding Window 방식의 비효율성 극복하기 위한 것임
- Sliding Window: 이미지에서 물체를 찾기 위해 window의 크기 및 비율을 임의로 바꿔가면서 모든 영역에 대해서 탐색하는 것
- 임의의 크기, 비율로 모든 영역을 탐색하는 것은 너무 느림
![image](https://github.com/user-attachments/assets/d5f4fee4-bd7b-4277-9ad0-5e09bd2e9981)


Selective search
- Sliding Window의 비효율성을 극복하기 위해 사용
- 색감, 질감, 영역크기 등을 이용해 non-object based segmentation 수행
- 이를 통해 많은 small segmented areas를 얻을 수 있음
- Bottom-up 방식으로 small segmented areas를 합쳐서 더 큰 area를 만듦
- 위의 작업을 반복하여 최종적으로 2,000개의 region proposal 생성
- CNN에 넣기 전에 같은 사이즈로 warp시켜야 함
- Selective search로 만든 bounding box는 정확하지 않기 때문에 물체를 정확히 감싸도록 조정해주는bounding box regression(선형회귀 모델)이 존재

![image](https://github.com/user-attachments/assets/5a2b143f-88e7-4966-9328-e3a66c618430)

---

CNN
- Warp 작업을 통해 region proposal의 결과를 224x224 크기로 CNN모델(AlexNet)에 넣음
- 4096차원의 feature vector에서 고정길이의 feature vector를 만들어 냄


![image](https://github.com/user-attachments/assets/864db990-5e2e-4d04-8248-219ede5f494f)

---

SVM
- CNN으로부터 feature가 추출되면 Linear SVM을 통해 classification 진행
- (softmax보다 SVM이 더 좋은 성능을 보여서 SVM 채택)
- SVM은 CNN으로부터 추출된 각각의 feature vector를 분류하는 역할
 
![image](https://github.com/user-attachments/assets/ececa380-bc7f-4b63-9749-75b03d00f47b)


---

**R-CNN의 과정**
1.  R-CNN은 selective search를 통해 region proposal을 먼저 뽑아낸 후 CNN 모델에 들어감.
2. CNN을 통해 feature vector를 뽑고 각각의 class마다 SVM로 classification을 수행.
3. localization error를 줄이기 위해 CNN feature를 이용하여 bounding box regression model을 수정.

![image](https://github.com/user-attachments/assets/a6d0c511-4dd1-408b-8344-ef91895347d3)

**R-CNN의 단점**
- selective search로 2000개의 region proposal을 뽑고 각 영역마다 CNN을 수행하기 때문에 CNN연산 * 2000 만큼의 시간이 걸려 수행시간이 매우 느림.
- CNN, SVM, Bounding Box Regression 총 세가지의 모델이 multi-stage pipelines으로 한 번에 학습되지 않음.
- 각 region proposal 에 대해 ConvNet forward pass를 실행할때 연산을 공유하지 않아 end-to-end 로 학습할 수 없음.
