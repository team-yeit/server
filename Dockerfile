# YOLO 모델 다운로드 단계
FROM alpine:3.19 AS yolo-models
RUN apk add --no-cache wget curl
WORKDIR /models

# YOLO 모델 파일들 다운로드 (OpenCV DNN에서 사용)
RUN set -e && \
    echo "Downloading YOLO model files..." && \
    wget -q --timeout=300 --tries=5 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg && \
    wget -q --timeout=300 --tries=5 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names && \
    curl -L --max-time 900 --retry 5 --retry-delay 30 \
    -o yolov4.weights \
    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights && \
    # 파일 크기 검증 (약 245MB)
    FILESIZE=$(stat -c%s yolov4.weights) && \
    [ ${FILESIZE} -gt 240000000 ] && [ ${FILESIZE} -lt 260000000 ] && \
    echo "YOLO model files downloaded successfully"

# Go 애플리케이션 빌드 단계
FROM gocv/opencv:4.11.0 AS go-builder
WORKDIR /app

# Go 모듈 의존성 설치
COPY go.mod go.sum ./
RUN go mod download

# 소스 코드 복사 및 빌드
COPY . .
ENV CGO_ENABLED=1 
ENV GOOS=linux
RUN go build -ldflags="-s -w" -tags netgo -o ui-automation .

# 최종 실행 단계
FROM gocv/opencv:4.11.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

# 빌드된 애플리케이션 복사
COPY --from=go-builder /app/ui-automation .

# YOLO 모델 파일들 복사
RUN mkdir -p cfg data
COPY --from=yolo-models /models/yolov4.cfg ./cfg/yolov4.cfg
COPY --from=yolo-models /models/yolov4.weights ./yolov4.weights
COPY --from=yolo-models /models/coco.names ./coco.names

# 디렉토리 권한 설정
RUN mkdir -p /tmp/ui-automation /app/logs && \
    chmod 755 /tmp/ui-automation /app/logs && \
    groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app /tmp/ui-automation && \
    chmod +x ui-automation

USER appuser

# 환경 변수 설정
ENV GIN_MODE=release 
ENV TMPDIR=/tmp/ui-automation

EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["./ui-automation"]

# ==================== 빌드 & 사용 가이드 ====================
#
# 빌드:
# docker build -t ui-automation:yolo .
#
# 실행:
# docker run -p 8000:8000 -e OPENAI_API_KEY=your_key ui-automation:yolo
#
# 개발 모드:
# docker run -p 8000:8000 -v $(pwd):/app/src -e OPENAI_API_KEY=your_key ui-automation:yolo