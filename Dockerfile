FROM alpine:3.19 AS yolo-models
RUN apk add --no-cache wget curl
WORKDIR /models
RUN set -e && \
    wget -q --timeout=300 --tries=5 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg && \
    wget -q --timeout=300 --tries=5 \
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names && \
    curl -L --max-time 900 --retry 5 --retry-delay 30 \
    -o yolov4.weights \
    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights && \
    FILESIZE=$(stat -c%s yolov4.weights) && \
    [ ${FILESIZE} -gt 240000000 ] && [ ${FILESIZE} -lt 260000000 ]

FROM debian:bullseye-slim AS darknet-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential pkg-config make libopencv-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ( git clone https://github.com/AlexeyAB/darknet.git /darknet || \
      git -c http.sslVerify=false clone https://github.com/AlexeyAB/darknet.git /darknet || \
      git clone http://github.com/AlexeyAB/darknet.git /darknet || \
      git clone --depth 1 https://github.com/pjreddie/darknet.git /darknet )
WORKDIR /darknet
RUN sed -i 's/GPU=1/GPU=0/' Makefile && \
    sed -i 's/CUDNN=1/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile && \
    make clean && make -j$(nproc)

FROM gocv/opencv:4.11.0 AS go-builder
WORKDIR /app
COPY --from=darknet-builder /darknet/libdarknet.so /usr/local/lib/
COPY --from=darknet-builder /darknet/include/ /usr/local/include/darknet/
COPY --from=darknet-builder /darknet/src/ /usr/local/include/darknet/src/
RUN ln -sf /usr/local/include/darknet/darknet.h /usr/local/include/darknet.h && \
    mkdir -p /usr/local/lib/pkgconfig && \
    echo "prefix=/usr/local\nexec_prefix=\${prefix}\nlibdir=\${exec_prefix}/lib\nincludedir=\${prefix}/include\n\nName: darknet\nDescription: Darknet Neural Network Framework\nVersion: 1.0\nLibs: -L\${libdir} -ldarknet\nCflags: -I\${includedir} -I\${includedir}/darknet" > /usr/local/lib/pkgconfig/darknet.pc && \
    ldconfig
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ENV CGO_ENABLED=1 GOOS=linux 
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV CGO_CFLAGS="-I/usr/local/include -I/usr/local/include/darknet"
ENV CGO_LDFLAGS="-L/usr/local/lib -ldarknet"
RUN go build -ldflags="-s -w" -tags netgo -o ui-automation .

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libc6 libgcc-s1 libgomp1 libstdc++6 \
    libopencv-core4.5 libopencv-imgproc4.5 libopencv-imgcodecs4.5 \
    libopencv-videoio4.5 libopencv-highgui4.5 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean
WORKDIR /app
COPY --from=go-builder /app/ui-automation .
COPY --from=go-builder /usr/local/lib/libdarknet.so /usr/local/lib/
COPY --from=yolo-models /models/ ./models/
RUN ldconfig && \
    mkdir -p /tmp/ui-automation /app/logs && \
    chmod 755 /tmp/ui-automation /app/logs && \
    groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app /tmp/ui-automation && \
    chmod +x ui-automation
USER appuser
ENV GIN_MODE=release TMPDIR=/tmp/ui-automation
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["./ui-automation"]

# ==================== 빌드 & 사용 가이드 ====================

#  빌드 명령어:
#  sudo docker build -t ui-automation:latest .

#  실행:
#  docker run -p 8000:8000 -e OPENAI_API_KEY=your_key ui-automation:latest

#  캐시 활용한 빠른 재빌드:
#  docker build -t ui-automation:latest . --cache-from ui-automation:latest

#  개발 모드 (볼륨 마운트):
#  docker run -p 8000:8000 -v $(pwd):/app/src -e OPENAI_API_KEY=your_key ui-automation:latest

#  이미지 크기 확인:
#  docker images ui-automation:latest

#  빌드 캐시 정리:
#  docker builder prune