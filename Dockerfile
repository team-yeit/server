# Multi-stage build for UI automation with YOLO detection

# Stage 1: YOLO 모델 다운로드 (Alpine 경량 이미지 사용)
FROM alpine:3.19 as yolo-models
RUN apk add --no-cache wget curl

WORKDIR /models

# YOLO 모델 파일들 다운로드 (검증된 안정적인 링크들 사용)
# 여러 fallback 옵션으로 안정성 확보
RUN set -e && \
    # yolov4.cfg 다운로드
    wget -q --timeout=120 --tries=3 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg && \
    \
    # coco.names 다운로드  
    wget -q --timeout=120 --tries=3 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names && \
    \
    # yolov4.weights 다운로드 (SourceForge 미러 사용 - 245MB)
    # 주 다운로드 시도
    ( wget -q --timeout=300 --tries=2 \
      -O yolov4.weights \
      "https://sourceforge.net/projects/darknet-yolo.mirror/files/darknet_yolo_v4_pre/yolov4.weights/download" ) || \
    # fallback 1: 직접 GitHub 링크
    ( wget -q --timeout=300 --tries=2 \
      https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights ) || \
    # fallback 2: curl 사용
    ( curl -L --max-time 300 --retry 2 \
      -o yolov4.weights \
      "https://sourceforge.net/projects/darknet-yolo.mirror/files/darknet_yolo_v4_pre/yolov4.weights/download" ) && \
    \
    # 파일 크기 검증 (yolov4.weights는 약 245MB여야 함)
    echo "Verifying downloaded files..." && \
    ls -la && \
    [ -f yolov4.cfg ] && echo "✓ yolov4.cfg downloaded" && \
    [ -f coco.names ] && echo "✓ coco.names downloaded" && \
    [ -f yolov4.weights ] && echo "✓ yolov4.weights downloaded" && \
    # 최소 크기 확인 (240MB 이상이어야 함)
    [ $(stat -c%s yolov4.weights) -gt 251658240 ] && echo "✓ yolov4.weights size verified" || \
    (echo "❌ yolov4.weights size check failed" && exit 1)

# Stage 2: Darknet 빌드
FROM debian:bullseye-slim as darknet-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /darknet
RUN git clone https://github.com/AlexeyAB/darknet.git . && \
    # CPU 전용 빌드로 설정
    sed -i 's/GPU=1/GPU=0/' Makefile && \
    sed -i 's/CUDNN=1/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
    make

# Stage 3: Go 빌드 환경
FROM gocv/opencv:4.8.1 as go-builder
WORKDIR /app

# Go 모듈 파일들 복사 및 의존성 다운로드
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# 소스 코드 복사
COPY . .

# Go 애플리케이션 빌드
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o main .

# Stage 4: 최종 실행 이미지
FROM debian:bullseye-slim

# 필요한 런타임 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libc6 \
    libgcc-s1 \
    libgomp1 \
    libstdc++6 \
    # OpenCV 런타임 라이브러리들
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    libopencv-videoio4.5 \
    libopencv-highgui4.5 \
    libopencv-objdetect4.5 \
    libopencv-features2d4.5 \
    libopencv-calib3d4.5 \
    libopencv-dnn4.5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 작업 디렉토리 생성
WORKDIR /app

# 빌드된 바이너리 복사
COPY --from=go-builder /app/main .

# Darknet 실행 파일 복사
COPY --from=darknet-builder /darknet/darknet /usr/local/bin/

# YOLO 모델 파일들 복사
COPY --from=yolo-models /models/yolov4.cfg /app/models/
COPY --from=yolo-models /models/coco.names /app/models/
COPY --from=yolo-models /models/yolov4.weights /app/models/

# 설정 파일들 복사 (있다면)
COPY configs/ /app/configs/ 2>/dev/null || true

# 로그 디렉토리 생성
RUN mkdir -p /app/logs

# 포트 노출 (필요에 따라 수정)
EXPOSE 8080

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 실행 권한 설정
RUN chmod +x main

# 최종 실행 명령
CMD ["./main"]