# DeepFake_Detection
딥페이크 탐지 AI 모델 개발

# 주제
다양한 이미지(jpg, jpeg, png, jfif) 또는 동영상(mp4, mov) 데이터를 이미지 단위로 입력으로 받아, 해당 콘텐츠가 진짜(Real)인지 딥페이크(Fake)인지 분류하는 AI 모델을 개발

# 프로젝트 개요
딥페이크 탐지는 고성능 모델보다 **입력 프레임의 품질과 정보 밀도**가 성능 안정성에 더 큰 영향을 미친다고 판단   
모델 구조를 과도하게 복잡하게 만들기보다,얼굴 품질 기반 데이터 정제 전략을 통해 딥페이크 탐지 성능을 개선하는 것을 목표로 한다.  

# 사용 기술
- Python
- Numpy
- OpenCV
- Dlib
- Pytorch
- HuggingFace
- Transformer
- ViT

# 데이터 전처리 : 		 		   
1. Frame Sampling (Video)
 * 비디오의 전체 프레임 중 균등 샘플링      
 * 계산 비용과 정보 다양성의 균형을 고려   

2. Face Detection & Alignment            
  * dlib 기반 얼굴 검출
  * 5개 핵심 랜드마크(눈, 코, 입) 기반 얼굴 정렬
  * 다양한 얼굴 크기 대응을 위해 multi-scale crop 적용
- 선택 이유
  * 얼굴 정렬을 통해 pose/scale variation 감소
  * downstream 모델의 입력 분포 안정화
  * 
3. Quality-based Face Filtering        
  (1) Blur Score
   * Laplacian variance로 프레임 선명도 측정   
   * 블러가 심한 경우 artifact 정보 손실    
  (2) Face Area
   * 얼굴 영역이 클수록 고주파 정보 보존
   * 작은 얼굴은 정보 손실 가능성 증가
 
4. Best-K Face Selection      
   Blur와 Area를 정규화하여 종합 점수 계산    

  * score = α · blur_norm + β · area_norm    
  * 얼굴 크기에 따라 α, β를 adaptive하게 조절    
  * 상위 K개의 얼굴만 모델 입력으로 사용    
  
  Noise 프레임 제거 + 정보 밀도 집중    

5. 모델 구조   
 * Backbone: ViT (Vision Transformer)   
 * Pretrained deepfake detection 모델 활용   
 * RGB 얼굴 이미지 입력    
- 선택 이유
 * ViT는 글로벌 패턴 학습에 강점
 * 얼굴 전체 artifact 탐지에 적합

6. 추론 및 Aggregation 전략
- Frame-level Prediction
  * 각 얼굴 crop에 대해 Fake probability 예측
- Video-level Aggregation
  * 단순 mean 대신 Top-K Mean Aggregation
- 이유
  * 딥페이크 artifact는 일부 프레임에 집중되는 경우가 많음
  * 평균보다 이상치(artifact 강한 프레임)에 민감
   
# 실험 결과 및 인사이트

- 단순 평균 aggregation 대비 Top-K aggregation이 artifact가 강한 프레임에 더 민감하게 반응함을 확인
- Blur 및 Face Area 기반 프레임 필터링을 통해 예측 확률의 분산이 감소하고 추론 안정성이 향상됨
- FFT, Hard Negative Mining과 같은 추가 기법은 복잡도 대비 성능 개선이 제한적이라 판단하여 제외
  
