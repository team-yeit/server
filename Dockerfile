# ==================== 멀티스테이지 빌드 ====================

# Build Stage - Go 컴파일 환경
FROM golang:1.21-bullseye AS builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (OpenCV 빌드용)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenCV 소스 다운로드 및 빌드 (최신 안정 버전)
RUN cd /tmp && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.11.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir -p opencv-4.11.0/build && \
    cd opencv-4.11.0/build && \
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-4.11.0/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_SHARED_LIBS=OFF \
        -D WITH_TBB=ON \
        -D WITH_V4L=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_python=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/opencv* 

# PKG_CONFIG_PATH 설정
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Go 모듈 초기화
COPY go.mod go.sum ./
RUN go mod download

# 소스코드 복사
COPY main.go .

# CGO 활성화 및 정적 링킹 설정
ENV CGO_ENABLED=1
ENV GOOS=linux

# Go 애플리케이션 빌드
RUN go build \
    -ldflags="-s -w -extldflags '-static'" \
    -a -installsuffix cgo \
    -o ui-automation \
    main.go

# ==================== Runtime Stage ====================

FROM ubuntu:22.04 AS runtime

# 필수 런타임 라이브러리 설치
RUN apt-get update && apt-get install -y \
    ca-certificates \
    wget \
    curl \
    libgtk-3-0 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libdc1394-22 \
    libtbb2 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 빌드된 바이너리 복사
COPY --from=builder /app/ui-automation .

# OpenCV 라이브러리 복사
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include /usr/local/include
COPY --from=builder /usr/local/share /usr/local/share

# 라이브러리 경로 업데이트
RUN ldconfig

# YOLO 모델 파일들 다운로드 (검증된 URL 사용)
RUN mkdir -p /app/models && cd /app/models && \
    # YOLOv4 config 파일
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg && \
    # COCO class names
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names && \
    # YOLOv4 weights (약 245MB) - 검증된 URL
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# YOLO 모델 파일 심볼릭 링크 생성 (현재 디렉토리에서 접근 가능하도록)
RUN ln -s /app/models/yolov4.cfg ./yolov4.cfg && \
    ln -s /app/models/coco.names ./coco.names && \
    ln -s /app/models/yolov4.weights ./yolov4.weights

# 임시 디렉토리 생성 (이미지 처리용)
RUN mkdir -p /tmp/ui-automation && chmod 777 /tmp/ui-automation

# 비 root 사용자 생성 (보안)
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /tmp/ui-automation
USER appuser

# 환경변수 설정
ENV TMPDIR=/tmp/ui-automation
ENV CGO_ENABLED=1

# 포트 노출
EXPOSE 8000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령어
CMD ["./ui-automation"]

# ==================== 개발용 단계 (선택사항) ====================

# 개발용 이미지 (docker build --target development)
FROM builder AS development

WORKDIR /app

# 개발 도구 설치
RUN apt-get update && apt-get install -y \
    vim \
    tree \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Go 개발 도구
RUN go install github.com/air-verse/air@latest
RUN go install github.com/go-delve/delve/cmd/dlv@latest

# 소스코드 복사 (개발용)
COPY . .

# air 설정 파일
COPY .air.toml .

# 개발 모드 실행 (hot reload)
CMD ["air"]

# ==================== 빌드 설명서 ====================

# 빌드 명령어들:
#
# 1. 프로덕션 빌드:
#    docker build -t ui-automation:latest .
#
# 2. 개발 빌드:
#    docker build --target development -t ui-automation:dev .
#
# 3. 실행:
#    docker run -p 8000:8000 -e OPENAI_API_KEY=your_key ui-automation:latest
#
# 4. 개발 모드 실행:
#    docker run -p 8000:8000 -v $(pwd):/app -e OPENAI_API_KEY=your_key ui-automation:dev
#
# 5. 도커 컴포즈 (docker-compose.yml 참고):
#    docker-compose up --build

# ==================== 최적화 팁 ====================

# 이미지 크기 최적화:
# - 멀티스테이지 빌드로 빌드 도구들 제거
# - 정적 링킹으로 라이브러리 의존성 최소화
# - 불필요한 패키지 정리
#
# 보안 강화:
# - 비 root 사용자로 실행
# - 최소한의 권한만 부여
# - 정기적인 베이스 이미지 업데이트
#
# 성능 최적화:
# - OpenCV 정적 링킹
# - YOLO 모델 로컬 캐싱
# - 멀티코어 빌드 활용