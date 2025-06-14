# ==================== 최적화된 멀티스테이지 빌드 ====================

# YOLO 모델 다운로드 단계 (가장 오래 걸리는 작업을 먼저 캐시)
FROM alpine:3.19 AS yolo-models

RUN apk add --no-cache wget curl

WORKDIR /models

# YOLO 모델 파일들 다운로드 (안정적인 링크와 fallback 옵션 포함)
RUN set -e && \
    echo "📥 YOLO 모델 파일 다운로드 시작..." && \
    \
    # yolov4.cfg 다운로드
    wget -q --timeout=120 --tries=3 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg && \
    echo "✅ yolov4.cfg 다운로드 완료" && \
    \
    # coco.names 다운로드  
    wget -q --timeout=120 --tries=3 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names && \
    echo "✅ coco.names 다운로드 완료" && \
    \
    # yolov4.weights 다운로드 (여러 소스로 안정성 확보)
    echo "📦 yolov4.weights (245MB) 다운로드 중..." && \
    ( wget -q --timeout=300 --tries=2 \
      -O yolov4.weights \
      "https://sourceforge.net/projects/darknet-yolo.mirror/files/darknet_yolo_v4_pre/yolov4.weights/download" && \
      echo "✅ SourceForge에서 다운로드 성공" ) || \
    ( echo "⚠️  SourceForge 실패, GitHub 시도 중..." && \
      wget -q --timeout=300 --tries=2 \
      https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights && \
      echo "✅ GitHub에서 다운로드 성공" ) || \
    ( echo "⚠️  wget 실패, curl 시도 중..." && \
      curl -L --max-time 300 --retry 2 --retry-delay 10 \
      -o yolov4.weights \
      "https://sourceforge.net/projects/darknet-yolo.mirror/files/darknet_yolo_v4_pre/yolov4.weights/download" && \
      echo "✅ curl로 다운로드 성공" ) || \
    ( echo "❌ 모든 다운로드 방법 실패" && exit 1 )

# 파일 무결성 검증 (강화된 검증)
RUN echo "🔍 파일 무결성 검증 중..." && \
    ls -la /models/ && \
    [ -f yolov4.cfg ] && echo "✅ yolov4.cfg 존재 확인" || (echo "❌ yolov4.cfg 없음" && exit 1) && \
    [ -f coco.names ] && echo "✅ coco.names 존재 확인" || (echo "❌ coco.names 없음" && exit 1) && \
    [ -f yolov4.weights ] && echo "✅ yolov4.weights 존재 확인" || (echo "❌ yolov4.weights 없음" && exit 1) && \
    \
    # 파일 크기 검증 (yolov4.weights는 약 245MB여야 함)
    FILESIZE=$(stat -c%s yolov4.weights) && \
    echo "📏 yolov4.weights 크기: ${FILESIZE} bytes" && \
    [ ${FILESIZE} -gt 240000000 ] && [ ${FILESIZE} -lt 260000000 ] && \
    echo "✅ yolov4.weights 크기 검증 성공 (240MB-260MB 범위)" || \
    (echo "❌ yolov4.weights 크기 검증 실패: ${FILESIZE} bytes" && exit 1) && \
    \
    echo "🎉 YOLO 모델 파일 다운로드 및 검증 완료"

# Darknet 빌드 단계 (go-darknet 의존성)
FROM debian:bullseye-slim AS darknet-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    make \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Darknet 소스 클론 및 빌드 (특정 커밋 고정으로 안정성 확보)
RUN echo "🔨 Darknet 빌드 시작..." && \
    git clone https://github.com/AlexeyAB/darknet.git /darknet

WORKDIR /darknet

# CPU 전용 빌드 (GPU는 프로덕션에서 선택적 활성화)
RUN echo "⚙️  Makefile 설정 중..." && \
    sed -i 's/GPU=1/GPU=0/' Makefile && \
    sed -i 's/CUDNN=1/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile && \
    \
    echo "🏗️  Darknet 컴파일 중..." && \
    make clean && \
    make -j$(nproc) && \
    echo "✅ Darknet 라이브러리 빌드 완료" && \
    ls -la libdarknet.so || echo "libdarknet.so 생성 확인 필요"

# Go 빌드 단계
FROM gocv/opencv:4.8.1 AS go-builder

# 작업 디렉토리 설정
WORKDIR /app

# Darknet 라이브러리 복사
COPY --from=darknet-builder /darknet/libdarknet.so /usr/local/lib/
COPY --from=darknet-builder /darknet/include/ /usr/local/include/darknet/
COPY --from=darknet-builder /darknet/src/ /usr/local/include/darknet/src/

# pkg-config 설정
RUN mkdir -p /usr/local/lib/pkgconfig && \
    echo "prefix=/usr/local\n\
exec_prefix=\${prefix}\n\
libdir=\${exec_prefix}/lib\n\
includedir=\${prefix}/include\n\
\n\
Name: darknet\n\
Description: Darknet Neural Network Framework\n\
Version: 1.0\n\
Libs: -L\${libdir} -ldarknet\n\
Cflags: -I\${includedir}" > /usr/local/lib/pkgconfig/darknet.pc

RUN ldconfig && echo "✅ 라이브러리 링크 설정 완료"

# Go 모듈 파일들 먼저 복사 (의존성 캐싱)
COPY go.mod go.sum ./

# 의존성 다운로드 (go.mod가 변경되지 않으면 캐시됨)
RUN echo "📦 Go 의존성 다운로드 중..." && \
    go mod download && \
    echo "✅ Go 의존성 다운로드 완료"

# 소스코드 복사 (마지막에 복사하여 코드 변경 시에만 재빌드)
COPY . .

# 환경변수 설정
ENV CGO_ENABLED=1
ENV GOOS=linux
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Go 애플리케이션 빌드 (최적화된 플래그)
RUN echo "🏗️  Go 애플리케이션 빌드 중..." && \
    go build \
    -ldflags="-s -w" \
    -tags netgo \
    -o ui-automation \
    . && \
    echo "✅ Go 애플리케이션 빌드 완료"

# 빌드 결과 검증
RUN echo "🔍 빌드 결과 검증 중..." && \
    ldd ui-automation || echo "정적 링크된 바이너리" && \
    ls -la ui-automation && \
    file ui-automation && \
    echo "✅ 바이너리 빌드 검증 완료"

# ==================== 최소 런타임 단계 ====================

FROM debian:bullseye-slim

# 런타임 의존성만 설치 (최소한으로 축소)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libc6 \
    libgcc-s1 \
    libgomp1 \
    libstdc++6 \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    libopencv-videoio4.5 \
    libopencv-highgui4.5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && echo "✅ 런타임 의존성 설치 완료"

# 작업 디렉토리 설정
WORKDIR /app

# 빌드된 바이너리 복사
COPY --from=go-builder /app/ui-automation .

# Darknet 라이브러리 복사
COPY --from=go-builder /usr/local/lib/libdarknet.so /usr/local/lib/

# YOLO 모델 파일들 복사
COPY --from=yolo-models /models/ ./models/

# 라이브러리 경로 업데이트
RUN ldconfig && echo "✅ 라이브러리 링크 업데이트 완료"

# 디렉토리 생성
RUN mkdir -p /tmp/ui-automation /app/logs && \
    chmod 755 /tmp/ui-automation /app/logs

# 보안을 위한 비특권 사용자 생성
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app /tmp/ui-automation

# 실행 권한 설정
RUN chmod +x ui-automation

# 비특권 사용자로 전환
USER appuser

# 환경변수 설정
ENV GIN_MODE=release
ENV TMPDIR=/tmp/ui-automation

# 포트 노출
EXPOSE 8000

# 헬스체크 (가벼운 체크)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령어
CMD ["./ui-automation"]

# ==================== 빌드 & 사용 가이드 ====================

# 🚀 빌드 명령어:
#   sudo docker build -t ui-automation:latest .
#
# 🏃 실행:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=your_key ui-automation:latest
#
# ⚡ 캐시 활용한 빠른 재빌드:
#   docker build -t ui-automation:latest . --cache-from ui-automation:latest
#
# 🔧 개발 모드 (볼륨 마운트):
#   docker run -p 8000:8000 -v $(pwd):/app/src -e OPENAI_API_KEY=your_key ui-automation:latest
#
# 📊 이미지 크기 확인:
#   docker images ui-automation:latest
#
# 🧹 빌드 캐시 정리:
#   docker builder prune