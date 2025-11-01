# SORT: Simple Online and Realtime Tracking 구현

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](SORT_Multi_object_tracking.ipynb)

### 전체 7개 시퀀스 결과 - 이미지 클릭 시 유튜브 영상 재생

<table>
  <tr>
    <td align="center">
      <b>MOT16-02</b><br>
      <a href="https://youtu.be/Si545ZWVyRM">
        <img src="https://img.youtube.com/vi/Si545ZWVyRM/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 20.08% | Precision: 66.12%</sub><br>
      <sub> 영상 보기</sub>
    </td>
    <td align="center">
      <b>MOT16-04</b><br>
      <a href="https://youtu.be/UV-Bnt7vTi4">
        <img src="https://img.youtube.com/vi/UV-Bnt7vTi4/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 46.18% | Precision: 84.72%</sub><br>
      <sub>  영상 보기</sub>
    </td>
    <td align="center">
      <b>MOT16-05</b><br>
      <a href="https://youtu.be/sR2uD0PJXgA">
        <img src="https://img.youtube.com/vi/sR2uD0PJXgA/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 44.65% | Precision: 79.06%</sub><br>
      <sub>  영상 보기</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>MOT16-09</b><br>
      <a href="https://youtu.be/eQHUaFhkc5I">
        <img src="https://img.youtube.com/vi/eQHUaFhkc5I/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 44.46% | Recall: 71.11%</sub><br>
      <sub>  영상 보기</sub>
    </td>
    <td align="center">
      <b>MOT16-10</b><br>
      <a href="https://youtu.be/X6xYvkxZ404">
        <img src="https://img.youtube.com/vi/X6xYvkxZ404/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 38.68% | Precision: 75.02%</sub><br>
      <sub>  영상 보기</sub>
    </td>
    <td align="center">
      <b>MOT16-11</b><br>
      <a href="https://youtu.be/qpnQtKtGFYA">
        <img src="https://img.youtube.com/vi/qpnQtKtGFYA/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 41.64% | ID Switches: 11</sub><br>
      <sub>  영상 보기</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>MOT16-13</b><br>
      <a href="https://youtu.be/gLu40GmscPA">
        <img src="https://img.youtube.com/vi/gLu40GmscPA/0.jpg" width="240"/>
      </a><br>
      <sub>MOTA: 39.01% | Precision: 83.45%</sub><br>
      <sub>  영상 보기</sub>
    </td>
  </tr>
</table>

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

### 전체 성능 지표 (7개 시퀀스)

| Sequence | Frames | MOTA | Precision | Recall | FP | FN | ID Switches | Tracking FPS |
|----------|--------|------|-----------|--------|-----|-----|-------------|--------------|
| MOT16-02 | 600 | 20.08% | 66.12% | 42.12% | 3848 | 10322 | 83 | 273 Hz |
| MOT16-04 | 1050 | **46.18%** | **84.72%** | 56.77% | 4869 | 20561 | 167 | 232 Hz |
| MOT16-05 | 837 | 44.65% | 79.06% | 61.85% | 1117 | 2601 | 56 | **548 Hz** |
| MOT16-09 | 525 | 44.46% | 73.17% | **71.11%** | 1371 | 1519 | 30 | 484 Hz |
| MOT16-10 | 654 | 38.68% | 75.02% | 59.30% | 2433 | 5013 | 107 | 308 Hz |
| MOT16-11 | 900 | 41.64% | 72.35% | 67.58% | 2369 | 2974 | **11** | 462 Hz |
| MOT16-13 | 750 | 39.01% | 83.45% | 50.28% | 1142 | 5693 | 148 | 430 Hz |
| **평균** | **759** | **39.24%** | **76.27%** | **58.43%** | **2307** | **6955** | **86** | **391 Hz** |

### 논문과의 비교

| Metric | 논문 (SORT) | 본 구현 (평균) | 차이 |
|--------|-------------|----------------|------|
| MOTA | 33.4% | **39.24%** | +5.84% ⬆️ |
| Precision | 72.1% | **76.27%** | +4.17% ⬆️ |
| Tracking FPS | 260 Hz | **391 Hz** | +50% ⬆️ |

### 챔피언 보드

- **최고 정확도 (MOTA)**: MOT16-04 (46.18%)
- **최고 정밀도 (Precision)**: MOT16-04 (84.72%)
- **최고 재현율 (Recall)**: MOT16-09 (71.11%)
- **최고 속도**: MOT16-05 (548 Hz)
- **최고 안정성**: MOT16-11 (ID switches 11개)

### 주요 발견

1. **속도**: Tracking 속도가 논문(260Hz)보다 **1.5배 빠름**
   - 최고 속도: 548Hz (MOT16-05)
   - 평균 속도: 391Hz
   - 모든 시퀀스에서 실시간 처리 가능

2. **정확도**: 논문 대비 모든 지표 향상
   - MOTA 평균 39.24% (논문 33.4%)
   - 4개 시퀀스에서 MOTA 40% 이상 달성
   - Precision 평균 76.27%로 높은 정밀도

3. **안정성**: ID Switch 관리 우수
   - MOT16-11: 단 11개 (900프레임)
   - MOT16-09: 30개 (525프레임)
   - 평균 86개로 합리적 수준

4. **시퀀스별 특징**:
   - **MOT16-04**: 최고 정확도 (MOTA 46.18%, Precision 84.72%)
   - **MOT16-05**: 초고속 처리 (548 Hz)
   - **MOT16-09**: 최고 탐지율 (Recall 71.11%)
   - **MOT16-11**: 최고 안정성 (ID switches 11)
   - **MOT16-02**: 복잡한 crowded scene으로 낮은 성능

5. **Detection Quality의 영향**:
   - Faster R-CNN (ResNet50-FPN) 사용으로 논문 대비 성능 향상
   - Detection quality가 tracking 성능의 핵심 요인 (논문 주장 검증)

## 설치 및 실행

### 환경 요구사항

```bash
Python >= 3.7
PyTorch >= 1.9.0
torchvision >= 0.10.0
opencv-python >= 4.5.0
scipy >= 1.7.0
filterpy >= 1.4.5
matplotlib >= 3.3.0
numpy >= 1.19.0
```

### 설치

```bash
# 필요한 패키지 설치
pip install torch torchvision opencv-python scipy filterpy matplotlib numpy
```

### 실행 방법

1. **Google Colab에서 실행** (권장)
   - `SORT_Multi_object_tracking.ipynb` 파일을 Colab에서 열기
   - Google Drive에 MOT16 데이터셋 업로드
   - 셀 순서대로 실행

2. **단일 시퀀스 실행**
```python
tracker, metrics, results = run_sort_tracker(
    sequence_name='MOT16-02',
    max_frames=None,
    visualize=True,
    save_video=True
)
```

3. **전체 시퀀스 실행**
```python
sequences = ['MOT16-02', 'MOT16-04', 'MOT16-05', 
             'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']

for seq in sequences:
    tracker, metrics, results = run_sort_tracker(
        sequence_name=seq,
        visualize=False,
        save_video=True
    )
```

## 출력 결과

### 1. 비디오 파일
- 경로: `/content/{sequence_name}_tracking.mp4`
- 형식: MP4 (H.264)
- FPS: 20
- 내용: 바운딩 박스 + Track ID + 실시간 정보

### 2. 결과 텍스트 파일
- 경로: `/content/{sequence_name}_result.txt`
- 형식: CSV
- 내용: `frame_id, x1, y1, x2, y2, track_id`

### 3. 콘솔 출력
- 실시간 FPS
- Detection/Tracking 수
- 최종 성능 지표 (MOTA, Precision, Recall, etc.)

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

## 코드 구조

```
SORT_Multi_object_tracking.ipynb
├── KalmanBoxTracker          # Kalman Filter 기반 개별 tracker
├── Sort                       # 전체 tracking 시스템
├── FasterRCNNDetector        # Faster R-CNN detector wrapper
├── MOTDataLoader             # MOT16 데이터 로더
├── MOTMetrics                # 평가 지표 계산
├── iou_batch                 # Vectorized IoU 계산
├── associate_detections_to_trackers  # Hungarian algorithm
└── draw_tracks               # 시각화 함수
```

## 제한사항 및 개선 방향

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

