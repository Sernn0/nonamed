# FontByMe - 내글씨폰트

HNU 2025 AI-programming course fall

20222328 윤여명 / 20202474 황경준 / 20232708 김수인

## 프로젝트 개요
- 사용자가 작성한 손글씨 PDF를 받아 글립 이미지를 추출하고, 벡터(SVG)·폰트(TTF)로 변환하는 파이프라인입니다.
- content/style 분리 모델(오토인코더 기반)과 벡터라이즈(potrace)·폰트 빌드(fontmake/fontTools)를 조합해 “내 글씨체”를 생성합니다.
- 로컬 Tkinter UI(`ui/font_ui.py`)를 통해 간편/정밀 모드 양식 다운로드 → PDF 업로드 → 결과물 다운로드(선택적으로 SVG/TTF) 흐름을 제공합니다.

## 실행 방법
### UI 실행
```bash
python run
```
- 최초 실행 시 프리플라이트 로그로 필수 의존성을 검사합니다.
- 양식 다운로드: 각 모드 우측 버튼을 눌러 다운로드합니다.
- PDF 업로드 시 페이지별 중앙 512x512를 잘라 256x256으로 리사이즈하고, charset 순서에 따라 `####_CODE.png`로 저장합니다.

### 학습 스크립트
- 스타일 인코더 학습:
```bash
python -m src.trainers.train_style_encoder \
  --train_index <train_json> --val_index <val_json> \
  --root <handwriting_raw/resizing> \
  --batch_size 32 --epochs 10 --style_dim 32 \
  --out_dir runs/style_encoder_full
```
- 조인트 학습:
```bash
python -m src.train.train_joint \
  --train_index <train_json> --val_index <val_json> \
  --root <handwriting_raw/resizing> \
  --content_latents runs/autoenc/content_latents.npy \
  --style_encoder_path runs/style_encoder/encoder_style_backbone.h5 \
  --batch_size 16 --epochs 1 --style_dim 32 --content_dim 64
```

## 개발 과정 / 아키텍처
1) **데이터 전처리**
   - `src/data/style_dataset.py`: JSON index → tf.data.Datasets (이미지 0~1 정규화, 256x256 리사이즈).
   - `src/data/preprocess_handwriting.py`: 손글씨 크롭/패딩/리사이즈.
   - `charset_50.txt`, `charset_220.txt`, `charset_2350.txt`: 문자 집합 정의.
2) **모델**
   - `src/models/style_encoder.py`: CNN + L2 normalize.
   - `src/models/decoder.py`: content+style → ConvTranspose 업샘플러, 출력 sigmoid(0~1).
   - content encoder는 오토인코더로 별도 학습(가중치 `.h5`, latent `.npy` 활용).
3) **학습 스크립트**
   - `src/trainers/train_style_encoder.py`: writer 분류로 스타일 백본 학습, remap으로 연속 라벨 처리.
   - `src/train/train_joint.py`: content_latents 고정, style encoder+decoder joint 재구성 학습.
4) **벡터라이즈·폰트 생성**
   - SVG: potrace (CLI) → `output.svg`
   - TTF: fontmake/fontTools → `output.ttf`
   - 현재 UI에는 TODO 더미 로직으로 자리만 잡혀 있음.
5) **UI**
   - `ui/font_ui.py`: 양식 다운로드 → PDF 드롭 → 로딩 → 결과 다운로드. Drag&Drop은 `tkinterdnd2` 설치 시 활성화.

## 설치 가이드

### 1. Python 패키지 설치 (공통)
```bash
pip install -r requirements-ui.txt
```

### 2. 시스템 도구 설치

#### macOS (Homebrew)
```bash
# Homebrew 설치 (미설치 시)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 필수 도구 설치
brew install poppler potrace fontforge
```

#### Windows

1. **Poppler** (PDF 처리용)
   - [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) 다운로드
   - 압축 해제 후 `bin` 폴더 경로를 환경변수 `POPPLER_PATH`에 추가
   - 또는 시스템 PATH에 추가

2. **Potrace** (PNG→SVG 변환용)
   - [potrace 다운로드](https://potrace.sourceforge.net/#downloading)
   - `potrace.exe`를 시스템 PATH에 추가

3. **FontForge** (SVG→TTF 변환용)
   - [FontForge 다운로드](https://fontforge.org/en-US/downloads/)
   - 설치 후 FontForge 설치 경로를 시스템 PATH에 추가

### 3. 실행
```bash
python run
```

## 기타
- 필수 의존성: `Pillow`, `numpy`, `pdf2image`, `fonttools`, `tensorflow`, `tkinterdnd2`
- 시스템 도구: Poppler, potrace, FontForge
