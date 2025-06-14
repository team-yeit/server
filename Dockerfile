# ==================== 최적화된 멀티스테이지 빌드 ====================

# YOLO 모델 다운로드 단계 (가장 오래 걸리는 작업을 먼저 캐시)
FROM alpine:3.19 AS yolo-models

RUN apk add --no-cache wget

WORKDIR /models

# YOLO 모델 파일들 다운로드 (이 단계가 캐시되면 재빌드 시 스킵됨)
RUN wget -q https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg && \
    wget -q https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names && \
    wget -q https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# 파일 무결성 검증 (선택사항)
RUN ls -la /models/ && \
    test -f yolov4.cfg && test -f coco.names && test -f yolov4.weights && \
    echo "✅ YOLO 모델 파일 다운로드 완료"

# Darknet 빌드 단계 (go-darknet 의존성)
FROM debian:bullseye-slim AS darknet-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    make \
    && rm -rf /var/lib/apt/lists/*

# Darknet 소스 클론 및 빌드 (특정 커밋 고정으로 안정성 확보)
RUN git clone https://github.com/AlexeyAB/darknet.git /darknet
WORKDIR /darknet

# CPU 전용 빌드 (GPU는 프로덕션에서 선택적 활성화)
RUN make clean && \
    make -j$(nproc) LIBSO=1 && \
    echo "✅ Darknet 라이브러리 빌드 완료"

FROM gocv/opencv:4.8.1 AS go-builder

# 작업 디렉토리 설정
WORKDIR /app

# Darknet 라이브러리 복사
COPY --from=darknet-builder /darknet/libdarknet.so /usr/local/lib/
COPY --from=darknet-builder /darknet/include/ /usr/local/include/darknet/
COPY --from=darknet-builder /darknet/src/ /usr/local/include/darknet/src/

# pkg-config 설정
RUN echo "prefix=/usr/local\n\
exec_prefix=\${prefix}\n\
libdir=\${exec_prefix}/lib\n\
includedir=\${prefix}/include\n\
\n\
Name: darknet\n\
Description: Darknet Neural Network Framework\n\
Version: 1.0\n\
Libs: -L\${libdir} -ldarknet\n\
Cflags: -I\${includedir}" > /usr/local/lib/pkgconfig/darknet.pc

RUN ldconfig

# Go 모듈 파일들 먼저 복사 (의존성 캐싱)
COPY go.mod go.sum ./

# 의존성 다운로드 (go.mod가 변경되지 않으면 캐시됨)
RUN go mod download && \
    echo "✅ Go 의존성 다운로드 완료"

# 소스코드 복사 (마지막에 복사하여 코드 변경 시에만 재빌드)
COPY main.go .

# 환경변수 설정
ENV CGO_ENABLED=1
ENV GOOS=linux
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Go 애플리케이션 빌드 (최적화된 플래그)
RUN go build \
    -ldflags="-s -w -linkmode external -extldflags '-static-libgcc'" \
    -tags netgo \
    -o ui-automation \
    main.go && \
    echo "✅ Go 애플리케이션 빌드 완료"

# 빌드 결과 검증
RUN ldd ui-automation && \
    ./ui-automation --help || echo "바이너리 빌드 성공" && \
    ls -la ui-automation

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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# 빌드된 바이너리 복사
COPY --from=go-builder /app/ui-automation .

# 필요한 라이브러리만 선별 복사
COPY --from=go-builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=go-builder /usr/local/lib/libdarknet.so /usr/local/lib/

# OpenCV 추가 의존 라이브러리 (필요시)
COPY --from=go-builder /usr/local/lib/pkgconfig/ /usr/local/lib/pkgconfig/

# YOLO 모델 파일들 복사
COPY --from=yolo-models /models/ ./

# 라이브러리 경로 업데이트
RUN ldconfig

# 임시 디렉토리 생성 (이미지 처리용)
RUN mkdir -p /tmp/ui-automation && chmod 755 /tmp/ui-automation

# 보안을 위한 비특권 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app /tmp/ui-automation
USER appuser

# 환경변수 설정
ENV GIN_MODE=release
ENV TMPDIR=/tmp/ui-automation
ENV CGO_ENABLED=0

# 포트 노출
EXPOSE 8000

# 헬스체크 (가벼운 체크)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령어
CMD ["./ui-automation"]

# ==================== 빌드 & 사용 가이드 ====================

# 🚀 빌드 명령어:
#   docker build -t ui-automation:latest .
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