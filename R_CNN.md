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
---

이미지 처리에서 그래프 방식을 활용할 때 사용하는 용어
<img width="542" alt="image" src="https://github.com/user-attachments/assets/6360db02-3af2-46ac-98ed-924d6f540631" />
<br>



