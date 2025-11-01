# SORT: Simple Online and Realtime Tracking 구현

## 논문 정보

본 저장소는 다음 논문의 PyTorch 구현입니다:

**제목**: Simple Online and Realtime Tracking (SORT)

**저자**: 
- Alex Bewley (Queensland University of Technology)
- Zongyuan Ge (Queensland University of Technology)
- Lionel Ott (University of Sydney)
- Fabio Ramos (University of Sydney)
- Ben Upcroft (Queensland University of Technology)

**발표**: 2016 IEEE International Conference on Image Processing (ICIP 2016)

**원본 논문**: [IEEE Xplore](https://ieeexplore.ieee.org/document/7533003)

## 개요

SORT는 Multiple Object Tracking (MOT) 문제를 해결하기 위한 실용적이고 효율적인 알고리즘입니다. 복잡한 appearance feature나 re-identification 없이 오직 **Kalman Filter**와 **Hungarian Algorithm**만을 사용하여 state-of-the-art 성능에 근접하면서도 260Hz의 빠른 속도를 달성합니다.

### 핵심 특징

- **Simple**: 최소한의 구성 요소만 사용 (Kalman Filter + Hungarian Algorithm)
- **Fast**: 260Hz 이상의 처리 속도로 실시간 추적 가능
- **Effective**: Detection quality에 집중하여 높은 정확도 달성
- **Online**: 과거와 현재 프레임만 사용하는 온라인 추적

## 모델 구조

### 1. Detection
- **Detector**: Faster R-CNN (ResNet50-FPN backbone)
- **Pre-trained**: COCO dataset
- **Target Class**: Person (class_id = 1)
- **Confidence Threshold**: 0.5

### 2. Tracking Components

#### Kalman Filter
- **State Vector** (7차원):
  ```
  x = [u, v, s, r, u̇, v̇, ṡ]ᵀ
  ```
  - u, v: 바운딩 박스 중심 좌표
  - s: 면적 (scale)
  - r: 종횡비 (aspect ratio, 상수)
  - u̇, v̇, ṡ: 속도 성분

- **Motion Model**: Constant Velocity Model
- **Prediction**: 선형 속도 모델로 다음 프레임 위치 예측
- **Update**: Detection과 매칭 시 Kalman update 수행

#### Data Association
- **Metric**: IoU (Intersection over Union) distance
- **Algorithm**: Hungarian Algorithm (최적 할당 문제 해결)
- **Threshold**: IoU ≥ 0.3

### 3. Track Management
- **Track Creation**: 새로운 detection이 기존 track과 매칭되지 않을 때 생성
- **Track Deletion**: Detection 없이 1프레임 경과 시 삭제 (TLost = 1)
- **Probationary Period**: min_hits = 3 (3번 이상 detection과 매칭되어야 유효한 track)

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `max_age` | 1 | Track이 detection 없이 유지되는 최대 프레임 수 |
| `min_hits` | 3 | Track이 출력되기 위한 최소 hit 수 |
| `iou_threshold` | 0.3 | Detection-Track 매칭을 위한 최소 IoU |
| `confidence_threshold` | 0.5 | Detection confidence 임계값 |

### Kalman Filter 파라미터

```python
# Measurement uncertainty
kf.R[2:,2:] *= 10.0

# Initial state uncertainty  
kf.P[4:,4:] *= 1000.0  # 속도 불확실성
kf.P *= 10.0

# Process uncertainty
kf.Q[-1,-1] *= 0.01
kf.Q[4:,4:] *= 0.01
```

## 데이터셋

### MOT16 Benchmark

**사용 시퀀스** (Training set):
- MOT16-02
- MOT16-04
- MOT16-05
- MOT16-09
- MOT16-10
- MOT16-11
- MOT16-13

**데이터셋 구조**:
```
MOT16/
└── train/
    ├── MOT16-02/
    │   ├── img1/        # 이미지 프레임
    │   └── gt/          # Ground truth
    │       └── gt.txt
    ├── MOT16-04/
    ...
```

**다운로드**: [MOT Challenge](https://motchallenge.net/data/MOT16/)

## 평가 결과

### 성능 지표

| Sequence | MOTA | Precision | Recall | FP | FN | ID Switches | Tracking FPS |
|----------|------|-----------|--------|-----|-----|-------------|--------------|
| MOT16-02 | 20.08% | 66.12% | 42.12% | 3848 | 10322 | 83 | 311.31 Hz |
| MOT16-04 | 46.18% | 84.72% | 56.77% | 4869 | 20561 | 167 | 232.33 Hz |
| MOT16-05 | 44.65% | 79.06% | 61.85% | 1117 | 2601 | 56 | 640.63 Hz |

### 논문과의 비교

| Metric | 논문 (SORT) | 본 구현 (평균) |
|--------|-------------|----------------|
| MOTA | 33.4% | 36.97% |
| Precision | 72.1% | 76.63% |
| Tracking FPS | 260 Hz | 394.76 Hz |

### 출력 영상


### 주요 발견

1. **속도**: Tracking 속도가 논문(260Hz)보다 1.5배 이상 빠름
   - 최고 속도: 640Hz (MOT16-05)
   - 평균 속도: 395Hz

2. **정확도**: 
   - MOT16-04, MOT16-05에서 논문 결과 초과 (MOTA 44-46%)
   - Precision 평균 76.63%로 높은 정밀도

3. **ID Switches**: 56-167개로 안정적인 추적 성능

4. **시퀀스별 차이**:
   - MOT16-05: 가장 균형잡힌 성능 (낮은 해상도, 명확한 장면)
   - MOT16-04: 가장 높은 MOTA (46.18%)
   - MOT16-02: 낮은 성능 (복잡한 장면, 높은 occlusion)


## 구현 세부사항

### 1. Detection Quality의 중요성

논문에서 강조한 대로, detection quality가 tracking performance에 결정적 영향을 미칩니다:

- ACF detector → Faster R-CNN 변경 시: **18.9% 성능 향상**
- 본 구현은 Faster R-CNN (ResNet50-FPN) 사용

### 2. 온라인 추적의 제약

- **TLost = 1**: Detection 없으면 즉시 track 삭제
- Long-term occlusion 처리 안 함
- Re-identification 없음
- 프레임 간 association에만 집중

### 3. 효율성 최적화

- Vectorized IoU 계산
- Hungarian algorithm의 최적 구현
- GPU 기반 detection
- 불필요한 feature extraction 제거

### 현재 제한사항

1. **Short-term tracking에만 최적화**: TLost=1로 long-term occlusion 처리 불가
2. **Re-identification 없음**: 동일 객체가 사라졌다 나타나면 새로운 ID 부여
3. **Appearance feature 미사용**: 순수 motion 기반 추적
4. **카메라 모션 보정 없음**: Static/moving camera 구분 없음

### 개선 방향

1. **Detection 개선**:
   - YOLOv8/YOLOv9 등 최신 detector 적용
   - Confidence threshold 튜닝
   - Multi-scale detection

2. **Tracking 개선**:
   - Deep SORT: Appearance feature 추가
   - TLost 값 증가로 occlusion 대응
   - Camera motion compensation

3. **응용 분야 확장**:
   - 자율주행 차량 추적
   - 스포츠 선수 추적
   - 군중 분석

## 결론

본 구현은 SORT 논문의 핵심 철학인 "Simple but Effective"를 충실히 재현했습니다:

- **간결성**: Kalman Filter + Hungarian Algorithm만으로 구현
- **속도**: 논문보다 빠른 395Hz 평균 속도 (실시간 응용 가능)
- **효율성**: 복잡한 feature 없이 높은 정확도 달성
- **실용성**: MOT16 벤치마크에서 검증된 성능

SORT는 복잡한 딥러닝 기반 tracker의 대안으로, 속도가 중요한 실시간 애플리케이션에 적합한 baseline 방법을 제시합니다.



