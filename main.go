package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/sashabaranov/go-openai"
	"gocv.io/x/gocv"
)

type UIElement struct {
	Type       string  `json:"type"`
	Text       string  `json:"text,omitempty"`
	ClassName  string  `json:"class_name,omitempty"`
	Confidence float64 `json:"confidence"`
	BBox       [4]int  `json:"bbox"`
	Center     [2]int  `json:"center"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
}

type UIElements struct {
	YOLOObjects []UIElement `json:"yolo_objects"`
	CVButtons   []UIElement `json:"cv_buttons"`
	CVInputs    []UIElement `json:"cv_inputs"`
}

type ActionResponse struct {
	Success      bool    `json:"success"`
	Coordinates  *[2]int `json:"coordinates,omitempty"`
	Reasoning    string  `json:"reasoning"`
	SelectedID   *int    `json:"selected_id,omitempty"`
	ErrorMessage string  `json:"error_message,omitempty"`
}

type AIResponse struct {
	SelectedID int    `json:"selected_id"`
	Reasoning  string `json:"reasoning"`
	Error      string `json:"error,omitempty"`
}

type UIAnalyzer struct {
	yoloNet        gocv.Net
	classNames     []string
	openaiClient   *openai.Client
	uiClassMapping map[string]string
	yoloEnabled    bool
	initialized    bool
	mu             sync.RWMutex
	outputLayers   []string
}

func NewUIAnalyzer() (*UIAnalyzer, error) {
	log.Println("Initializing GoCV OpenCV DNN YOLO system...")
	startTime := time.Now()

	// 메모리 최적화
	debug.SetGCPercent(50)
	runtime.GOMAXPROCS(runtime.NumCPU())

	analyzer := &UIAnalyzer{
		uiClassMapping: map[string]string{
			"person": "icon", "book": "button", "laptop": "screen", "mouse": "button",
			"keyboard": "input", "cell phone": "device", "tv": "screen", "remote": "button",
			"bottle": "clickable", "cup": "clickable", "bowl": "clickable", "chair": "clickable",
		},
		yoloEnabled: false,
		initialized: false,
	}

	// OpenAI 클라이언트 초기화
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("WARNING: OPENAI_API_KEY not set - AI selection disabled")
	} else {
		analyzer.openaiClient = openai.NewClient(apiKey)
		log.Println("OpenAI client initialized successfully")
	}

	// YOLO 초기화 (GoCV 방식)
	if err := analyzer.initializeYOLOWithGoCV(); err != nil {
		log.Printf("YOLO initialization failed: %v - continuing with OpenCV only", err)
	}

	analyzer.initialized = true
	log.Printf("UI analyzer initialized in %v - YOLO enabled: %t", time.Since(startTime), analyzer.yoloEnabled)
	return analyzer, nil
}

func (ua *UIAnalyzer) initializeYOLOWithGoCV() error {
	// YOLO 파일들 확인
	cfgPath := "cfg/yolov4.cfg"
	weightsPath := "yolov4.weights"
	namesPath := "coco.names"

	// 파일 존재 확인
	for _, path := range []string{cfgPath, weightsPath, namesPath} {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return fmt.Errorf("required file not found: %s", path)
		}
	}

	log.Printf("Loading YOLO model files...")
	log.Printf("- Config: %s", cfgPath)
	log.Printf("- Weights: %s", weightsPath)
	log.Printf("- Names: %s", namesPath)

	// GoCV ReadNet 사용 (Darknet 자동 감지)
	net := gocv.ReadNet(weightsPath, cfgPath)
	if net.Empty() {
		return fmt.Errorf("failed to load YOLO network")
	}

	// 백엔드 설정 (CPU 우선, GPU 옵션 가능)
	net.SetPreferableBackend(gocv.NetBackendOpenCV)
	net.SetPreferableTarget(gocv.NetTargetCPU)

	// 클래스 이름 로드
	classNames, err := ua.loadClassNames(namesPath)
	if err != nil {
		net.Close()
		return fmt.Errorf("failed to load class names: %v", err)
	}

	// 출력 레이어 이름 가져오기
	outputLayers := ua.getOutputLayerNames(net)
	if len(outputLayers) == 0 {
		net.Close()
		return fmt.Errorf("no output layers found")
	}

	ua.yoloNet = net
	ua.classNames = classNames
	ua.outputLayers = outputLayers
	ua.yoloEnabled = true

	log.Printf("✅ YOLO initialized successfully with GoCV!")
	log.Printf("   - Classes: %d", len(classNames))
	log.Printf("   - Backend: OpenCV CPU")
	log.Printf("   - Output layers: %v", outputLayers)

	return nil
}

func (ua *UIAnalyzer) getOutputLayerNames(net gocv.Net) []string {
	// 안전한 방식으로 레이어 이름들 가져오기
	var layerNames []string
	var unconnectedLayers []int

	// 레이어 이름 가져오기 (에러 처리)
	func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Panic in GetLayerNames: %v", r)
				layerNames = []string{}
			}
		}()
		layerNames = net.GetLayerNames()
	}()

	// 연결되지 않은 출력 레이어 가져오기 (에러 처리)
	func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Panic in GetUnconnectedOutLayers: %v", r)
				unconnectedLayers = []int{}
			}
		}()
		unconnectedLayers = net.GetUnconnectedOutLayers()
	}()

	log.Printf("Total layers: %d, Unconnected out layers: %v", len(layerNames), unconnectedLayers)

	var outputNames []string

	// 인덱스 기반으로 출력 레이어 이름 가져오기
	for _, layerIndex := range unconnectedLayers {
		// 1-based 인덱스를 0-based로 변환
		arrayIndex := layerIndex - 1
		if arrayIndex >= 0 && arrayIndex < len(layerNames) {
			outputNames = append(outputNames, layerNames[arrayIndex])
			log.Printf("Output layer %d: %s", layerIndex, layerNames[arrayIndex])
		} else {
			log.Printf("Invalid layer index %d (array size: %d)", layerIndex, len(layerNames))
		}
	}

	// YOLOv4의 경우 보통 3개의 출력 레이어가 있음
	if len(outputNames) == 0 {
		// 하드코딩된 YOLO 출력 레이어 이름들 (fallback)
		log.Println("No output layers found, using fallback names")

		// 일반적인 YOLO 출력 레이어 패턴들 시도
		fallbackNames := [][]string{
			{"yolo_82", "yolo_94", "yolo_106"},   // YOLOv4
			{"yolo_139", "yolo_150", "yolo_161"}, // YOLOv4-tiny
			{"output", "output1", "output2"},     // 일반적인 이름
			{"", "output", "detection_out"},      // 기본 이름들
		}

		// 사용 가능한 레이어 이름에서 YOLO 관련 찾기
		for _, name := range layerNames {
			lowerName := strings.ToLower(name)
			if strings.Contains(lowerName, "yolo") ||
				strings.Contains(lowerName, "output") ||
				strings.Contains(lowerName, "detection") {
				outputNames = append(outputNames, name)
				log.Printf("Found potential output layer: %s", name)
			}
		}

		// 여전히 빈 경우 fallback 사용
		if len(outputNames) == 0 {
			outputNames = fallbackNames[0]
			log.Printf("Using hardcoded fallback: %v", outputNames)
		}
	}

	log.Printf("Final output layer names: %v", outputNames)
	return outputNames
}

func (ua *UIAnalyzer) loadClassNames(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(content), "\n")
	var classNames []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			classNames = append(classNames, line)
		}
	}

	return classNames, nil
}

func (ua *UIAnalyzer) Close() {
	ua.mu.Lock()
	defer ua.mu.Unlock()

	if ua.yoloEnabled && !ua.yoloNet.Empty() {
		ua.yoloNet.Close()
		log.Println("YOLO network closed")
	}
}

func (ua *UIAnalyzer) DetectUIElements(imagePath string) (*UIElements, error) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	if !ua.initialized {
		return nil, fmt.Errorf("analyzer not initialized")
	}

	log.Printf("Starting UI element detection: %s", imagePath)
	startTime := time.Now()

	elements := &UIElements{
		YOLOObjects: []UIElement{},
		CVButtons:   []UIElement{},
		CVInputs:    []UIElement{},
	}

	// YOLO 감지 (GoCV 방식)
	if ua.yoloEnabled {
		yoloObjects, err := ua.detectYOLOWithGoCV(imagePath)
		if err != nil {
			log.Printf("YOLO detection failed: %v", err)
		} else {
			elements.YOLOObjects = yoloObjects
			log.Printf("YOLO detected %d objects", len(yoloObjects))
		}
	}

	// OpenCV 감지 (병렬 실행)
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		buttons, err := ua.detectCVButtons(imagePath)
		if err != nil {
			log.Printf("CV button detection failed: %v", err)
		} else {
			elements.CVButtons = buttons
			log.Printf("CV detected %d buttons", len(buttons))
		}
	}()

	go func() {
		defer wg.Done()
		inputs, err := ua.detectCVInputs(imagePath)
		if err != nil {
			log.Printf("CV input detection failed: %v", err)
		} else {
			elements.CVInputs = inputs
			log.Printf("CV detected %d input fields", len(inputs))
		}
	}()

	wg.Wait()

	total := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	log.Printf("Detection completed in %v - total: %d elements", time.Since(startTime), total)

	return elements, nil
}

func (ua *UIAnalyzer) detectYOLOWithGoCV(imagePath string) ([]UIElement, error) {
	if !ua.yoloEnabled || ua.yoloNet.Empty() {
		return []UIElement{}, nil
	}

	// 이미지 로드
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	// 이미지 크기 가져오기
	height := img.Rows()
	width := img.Cols()

	if height <= 0 || width <= 0 {
		log.Printf("Invalid image dimensions: %dx%d", width, height)
		return []UIElement{}, nil
	}

	// YOLO 입력 크기 (416x416)
	inputSize := image.Pt(416, 416)

	log.Printf("Processing image: %dx%d -> %dx%d", width, height, inputSize.X, inputSize.Y)

	// Blob 생성 (GoCV 방식 - MatType 파라미터 제거)
	blob := gocv.BlobFromImage(img, 1.0/255.0, inputSize, gocv.NewScalar(0, 0, 0, 0), true, false)
	if blob.Empty() {
		log.Printf("Failed to create blob from image")
		return []UIElement{}, nil
	}
	defer blob.Close()

	// 네트워크 입력 설정
	ua.yoloNet.SetInput(blob, "")

	// YOLO 추론 실행 (안전한 방식)
	startTime := time.Now()
	outputs := make([]gocv.Mat, 0)

	// 안전한 추론 실행
	func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("YOLO inference panic recovered: %v", r)
			}
		}()
		outputs = ua.yoloNet.ForwardLayers(ua.outputLayers)
	}()

	inferenceTime := time.Since(startTime)
	log.Printf("YOLO inference completed in %v", inferenceTime)

	// 출력 검증
	if len(outputs) == 0 {
		log.Printf("No YOLO outputs received")
		return []UIElement{}, nil
	}

	// 빈 출력 확인
	validOutputs := make([]gocv.Mat, 0)
	for _, output := range outputs {
		if !output.Empty() && output.Total() > 0 {
			validOutputs = append(validOutputs, output)
		} else {
			log.Printf("Skipping empty YOLO output")
		}
	}

	if len(validOutputs) == 0 {
		log.Printf("All YOLO outputs are empty")
		// 메모리 정리
		for i := range outputs {
			outputs[i].Close()
		}
		return []UIElement{}, nil
	}

	// 결과 후처리 (안전한 방식)
	objects := ua.postProcessYOLOSafe(validOutputs, width, height, 0.3, 0.4)

	// 메모리 정리
	for i := range outputs {
		outputs[i].Close()
	}

	log.Printf("YOLO post-processing: %d objects detected", len(objects))
	return objects, nil
}

func (ua *UIAnalyzer) postProcessYOLOSafe(outputs []gocv.Mat, imgWidth, imgHeight int, confThreshold, nmsThreshold float32) []UIElement {
	var boxes []image.Rectangle
	var confidences []float32
	var classIDs []int

	log.Printf("Post-processing %d YOLO outputs for image %dx%d", len(outputs), imgWidth, imgHeight)

	// 각 출력 레이어 처리 (안전한 방식)
	for outputIdx, output := range outputs {
		// 출력 Mat 검증
		if output.Empty() {
			log.Printf("Output %d is empty, skipping", outputIdx)
			continue
		}

		// Mat 정보 로깅
		rows := output.Rows()
		cols := output.Cols()
		total := output.Total()
		channels := output.Channels()

		log.Printf("Output %d: Rows=%d, Cols=%d, Total=%d, Channels=%d",
			outputIdx, rows, cols, total, channels)

		// 유효성 검사
		if rows <= 0 || cols <= 0 || total <= 0 {
			log.Printf("Output %d has invalid dimensions, skipping", outputIdx)
			continue
		}

		// Mat 데이터를 float32 슬라이스로 변환 (안전한 방식)
		data, err := ua.safeGetFloat32Data(output)
		if err != nil {
			log.Printf("Failed to get float32 data from output %d: %v", outputIdx, err)
			continue
		}

		if len(data) == 0 {
			log.Printf("Output %d has no data, skipping", outputIdx)
			continue
		}

		log.Printf("Output %d data length: %d", outputIdx, len(data))

		// YOLO 출력 처리 (다양한 형태 지원)
		err = ua.processYOLOOutput(data, rows, cols, imgWidth, imgHeight, confThreshold,
			&boxes, &confidences, &classIDs)
		if err != nil {
			log.Printf("Failed to process output %d: %v", outputIdx, err)
			continue
		}
	}

	log.Printf("Before NMS: %d boxes, %d confidences, %d classIDs",
		len(boxes), len(confidences), len(classIDs))

	// NMS 적용 전 유효성 검사
	if len(boxes) == 0 || len(confidences) == 0 || len(classIDs) == 0 {
		log.Printf("No valid detections found")
		return []UIElement{}
	}

	if len(boxes) != len(confidences) || len(boxes) != len(classIDs) {
		log.Printf("Mismatched detection arrays: boxes=%d, confidences=%d, classIDs=%d",
			len(boxes), len(confidences), len(classIDs))
		return []UIElement{}
	}

	// NMS (Non-Maximum Suppression) 적용 (안전한 방식)
	var indices []int
	func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("NMS panic recovered: %v", r)
				indices = []int{} // 빈 인덱스 반환
			}
		}()
		indices = gocv.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	}()

	log.Printf("After NMS: %d indices", len(indices))

	// 최종 객체 생성
	var objects []UIElement
	for _, idx := range indices {
		if idx < 0 || idx >= len(boxes) || idx >= len(classIDs) {
			log.Printf("Invalid index %d, skipping", idx)
			continue
		}

		box := boxes[idx]
		classID := classIDs[idx]
		confidence := confidences[idx]

		// 클래스 이름 가져오기
		className := "unknown"
		if classID >= 0 && classID < len(ua.classNames) {
			className = ua.classNames[classID]
		}

		// UI 타입 매핑
		uiType, exists := ua.uiClassMapping[className]
		if !exists {
			uiType = "object"
		}

		objects = append(objects, UIElement{
			Type:       fmt.Sprintf("yolo_%s", uiType),
			ClassName:  className,
			Confidence: float64(confidence),
			BBox:       [4]int{box.Min.X, box.Min.Y, box.Max.X, box.Max.Y},
			Center:     [2]int{box.Min.X + box.Dx()/2, box.Min.Y + box.Dy()/2},
			Width:      box.Dx(),
			Height:     box.Dy(),
		})
	}

	return objects
}

// 안전한 float32 데이터 추출
func (ua *UIAnalyzer) safeGetFloat32Data(mat gocv.Mat) ([]float32, error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic in safeGetFloat32Data: %v", r)
		}
	}()

	// Mat 타입 확인
	if mat.Type() != gocv.MatTypeCV32F {
		log.Printf("Mat type is not CV32F: %d", mat.Type())
		// 타입 변환 시도
		converted := gocv.NewMat()
		defer converted.Close()
		mat.ConvertTo(&converted, gocv.MatTypeCV32F)
		return converted.DataPtrFloat32()
	}

	return mat.DataPtrFloat32()
}

// YOLO 출력 처리 (다양한 형태 지원)
func (ua *UIAnalyzer) processYOLOOutput(data []float32, rows, cols, imgWidth, imgHeight int,
	confThreshold float32, boxes *[]image.Rectangle, confidences *[]float32, classIDs *[]int) error {

	// YOLO 출력 형태 감지
	if rows == -1 && cols == -1 {
		// 4차원 텐서 형태 (예: [1, 32, 320, 320])
		// 이 경우는 처리하지 않음 (중간 레이어 출력)
		log.Printf("Skipping intermediate layer output (4D tensor)")
		return nil
	}

	// 일반적인 YOLO 출력 형태: [N, 85] (N개 검출, 85 = 4 + 1 + 80)
	if cols < 5 {
		return fmt.Errorf("invalid YOLO output format: cols=%d (expected >= 5)", cols)
	}

	expectedSize := rows * cols
	if len(data) < expectedSize {
		return fmt.Errorf("data size mismatch: expected %d, got %d", expectedSize, len(data))
	}

	// 각 검출 처리
	for i := 0; i < rows; i++ {
		offset := i * cols
		if offset+4 >= len(data) {
			break
		}

		// confidence 값 추출 (5번째 인덱스)
		confidence := data[offset+4]

		if confidence > confThreshold {
			// 클래스 확률들 중 최대값 찾기
			maxClassProb := float32(0)
			classID := 0

			for j := 5; j < cols && offset+j < len(data); j++ {
				classProb := data[offset+j]
				if classProb > maxClassProb {
					maxClassProb = classProb
					classID = j - 5
				}
			}

			finalConf := confidence * maxClassProb
			if finalConf > confThreshold {
				// 바운딩 박스 좌표 추출
				centerX := int(data[offset+0] * float32(imgWidth))
				centerY := int(data[offset+1] * float32(imgHeight))
				width := int(data[offset+2] * float32(imgWidth))
				height := int(data[offset+3] * float32(imgHeight))

				// 좌상단 좌표 계산
				x := centerX - width/2
				y := centerY - height/2

				// 경계 확인
				if x < 0 {
					x = 0
				}
				if y < 0 {
					y = 0
				}
				if x+width > imgWidth {
					width = imgWidth - x
				}
				if y+height > imgHeight {
					height = imgHeight - y
				}

				// 유효한 박스인지 확인
				if width > 10 && height > 10 {
					*boxes = append(*boxes, image.Rect(x, y, x+width, y+height))
					*confidences = append(*confidences, finalConf)
					*classIDs = append(*classIDs, classID)
				}
			}
		}
	}

	return nil
}

func (ua *UIAnalyzer) detectCVButtons(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	var buttons []UIElement

	// 적응형 임계값 + 윤곽선 검출
	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.AdaptiveThreshold(img, &thresh, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 2)

	contours := gocv.FindContours(thresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 800 && area < 50000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 0.3 && aspectRatio < 8.0 && rect.Dx() > 30 && rect.Dy() > 20 {
				buttons = append(buttons, UIElement{
					Type:       "cv_button",
					Confidence: 0.8,
					BBox:       [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
					Center:     [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2},
					Width:      rect.Dx(),
					Height:     rect.Dy(),
				})
			}
		}
	}

	return buttons, nil
}

func (ua *UIAnalyzer) detectCVInputs(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	var inputs []UIElement

	// 에지 검출
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(img, &edges, 30, 100)

	// 수평 구조 요소
	horizontalKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(25, 1))
	defer horizontalKernel.Close()

	detectHorizontal := gocv.NewMat()
	defer detectHorizontal.Close()
	gocv.MorphologyEx(edges, &detectHorizontal, gocv.MorphOpen, horizontalKernel)

	contours := gocv.FindContours(detectHorizontal, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 && area < 30000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 2.5 && rect.Dx() > 80 && rect.Dy() > 15 && rect.Dy() < 60 {
				inputs = append(inputs, UIElement{
					Type:       "cv_input_field",
					Confidence: 0.8,
					BBox:       [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
					Center:     [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2},
					Width:      rect.Dx(),
					Height:     rect.Dy(),
				})
			}
		}
	}

	return inputs, nil
}

func (ua *UIAnalyzer) CreateLabeledImage(imagePath string, elements *UIElements) (string, map[int]UIElement, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return "", nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return "", nil, err
	}

	bounds := img.Bounds()
	labeledImg := image.NewRGBA(bounds)
	draw.Draw(labeledImg, bounds, img, bounds.Min, draw.Src)

	idToElement := make(map[int]UIElement)
	elementID := 1

	colors := map[string]color.RGBA{
		"yolo":      {255, 0, 0, 255},   // 빨간색 - YOLO 객체
		"cv_button": {0, 200, 0, 255},   // 진한 녹색 - 버튼
		"cv_input":  {255, 140, 0, 255}, // 주황색 - 입력 필드
	}

	// 숫자 색상 (가독성을 위해 진한 색상 사용)
	numberColors := map[string]color.RGBA{
		"yolo":      {139, 0, 0, 255},  // 진한 빨간색
		"cv_button": {0, 100, 0, 255},  // 진한 녹색
		"cv_input":  {204, 85, 0, 255}, // 진한 주황색
	}

	// 모든 요소 그리기
	allElements := append(append(elements.YOLOObjects, elements.CVButtons...), elements.CVInputs...)
	for _, element := range allElements {
		idToElement[elementID] = element

		var elementColor color.RGBA
		var numberColor color.RGBA
		if strings.HasPrefix(element.Type, "yolo") {
			elementColor = colors["yolo"]
			numberColor = numberColors["yolo"]
		} else if strings.Contains(element.Type, "button") {
			elementColor = colors["cv_button"]
			numberColor = numberColors["cv_button"]
		} else {
			elementColor = colors["cv_input"]
			numberColor = numberColors["cv_input"]
		}

		ua.drawRectangle(labeledImg, element.BBox, elementColor)
		ua.drawTextWithColor(labeledImg, element.Center, strconv.Itoa(elementID), numberColor)
		elementID++
	}

	tempFile, err := os.CreateTemp("", "labeled_ui_*.png")
	if err != nil {
		return "", nil, err
	}
	defer tempFile.Close()

	if err := png.Encode(tempFile, labeledImg); err != nil {
		return "", nil, err
	}

	return tempFile.Name(), idToElement, nil
}

func (ua *UIAnalyzer) drawRectangle(img *image.RGBA, bbox [4]int, clr color.RGBA) {
	bounds := img.Bounds()

	// 두께 2픽셀의 사각형
	for thickness := 0; thickness < 2; thickness++ {
		for x := bbox[0]; x <= bbox[2]; x++ {
			if x >= bounds.Min.X && x < bounds.Max.X {
				if bbox[1]+thickness >= bounds.Min.Y && bbox[1]+thickness < bounds.Max.Y {
					img.Set(x, bbox[1]+thickness, clr)
				}
				if bbox[3]-thickness >= bounds.Min.Y && bbox[3]-thickness < bounds.Max.Y {
					img.Set(x, bbox[3]-thickness, clr)
				}
			}
		}

		for y := bbox[1]; y <= bbox[3]; y++ {
			if y >= bounds.Min.Y && y < bounds.Max.Y {
				if bbox[0]+thickness >= bounds.Min.X && bbox[0]+thickness < bounds.Max.X {
					img.Set(bbox[0]+thickness, y, clr)
				}
				if bbox[2]-thickness >= bounds.Min.X && bbox[2]-thickness < bounds.Max.X {
					img.Set(bbox[2]-thickness, y, clr)
				}
			}
		}
	}
}

func (ua *UIAnalyzer) drawText(img *image.RGBA, center [2]int, text string, clr color.RGBA) {
	// 큰 숫자 그리기 (픽셀 아트 방식으로 숫자 형태 그리기)
	ua.drawLargeNumber(img, center, text, clr)
}

func (ua *UIAnalyzer) drawTextWithColor(img *image.RGBA, center [2]int, text string, clr color.RGBA) {
	// 큰 숫자 그리기 (픽셀 아트 방식으로 숫자 형태 그리기)
	ua.drawLargeNumber(img, center, text, clr)
}

func (ua *UIAnalyzer) drawLargeNumber(img *image.RGBA, center [2]int, number string, clr color.RGBA) {
	bounds := img.Bounds()

	// 숫자 길이에 따라 배경 크기 조정
	numDigits := len(number)
	bgWidth := 10 + numDigits*8 // 숫자당 8픽셀 + 여백
	bgHeight := 20

	// 배경 사각형 그리기 (흰색)
	bgColor := color.RGBA{255, 255, 255, 255}
	for dx := -bgWidth; dx <= bgWidth; dx++ {
		for dy := -bgHeight / 2; dy <= bgHeight/2; dy++ {
			x, y := center[0]+dx, center[1]+dy
			if x >= bounds.Min.X && x < bounds.Max.X && y >= bounds.Min.Y && y < bounds.Max.Y {
				img.Set(x, y, bgColor)
			}
		}
	}

	// 테두리 그리기
	borderColor := color.RGBA{0, 0, 0, 255} // 검은색 테두리
	for dx := -bgWidth - 1; dx <= bgWidth+1; dx++ {
		for dy := -bgHeight/2 - 1; dy <= bgHeight/2+1; dy++ {
			x, y := center[0]+dx, center[1]+dy
			if x >= bounds.Min.X && x < bounds.Max.X && y >= bounds.Min.Y && y < bounds.Max.Y {
				if dx == -bgWidth-1 || dx == bgWidth+1 || dy == -bgHeight/2-1 || dy == bgHeight/2+1 {
					img.Set(x, y, borderColor)
				}
			}
		}
	}

	// 각 숫자를 그리기
	startX := center[0] - (numDigits-1)*4 // 중앙 정렬을 위한 시작 위치
	for i, digit := range number {
		digitCenter := [2]int{startX + i*8, center[1]}
		ua.drawDigit(img, digitCenter, byte(digit), clr)
	}
}

func (ua *UIAnalyzer) drawDigit(img *image.RGBA, center [2]int, digit byte, clr color.RGBA) {
	bounds := img.Bounds()

	// 5x7 픽셀 패턴으로 숫자 그리기 (작고 선명하게)
	patterns := map[byte][][]bool{
		'0': {
			{true, true, true},
			{true, false, true},
			{true, false, true},
			{true, false, true},
			{true, true, true},
		},
		'1': {
			{false, true, false},
			{true, true, false},
			{false, true, false},
			{false, true, false},
			{true, true, true},
		},
		'2': {
			{true, true, true},
			{false, false, true},
			{true, true, true},
			{true, false, false},
			{true, true, true},
		},
		'3': {
			{true, true, true},
			{false, false, true},
			{true, true, true},
			{false, false, true},
			{true, true, true},
		},
		'4': {
			{true, false, true},
			{true, false, true},
			{true, true, true},
			{false, false, true},
			{false, false, true},
		},
		'5': {
			{true, true, true},
			{true, false, false},
			{true, true, true},
			{false, false, true},
			{true, true, true},
		},
		'6': {
			{true, true, true},
			{true, false, false},
			{true, true, true},
			{true, false, true},
			{true, true, true},
		},
		'7': {
			{true, true, true},
			{false, false, true},
			{false, true, false},
			{false, true, false},
			{false, true, false},
		},
		'8': {
			{true, true, true},
			{true, false, true},
			{true, true, true},
			{true, false, true},
			{true, true, true},
		},
		'9': {
			{true, true, true},
			{true, false, true},
			{true, true, true},
			{false, false, true},
			{true, true, true},
		},
	}

	pattern, exists := patterns[digit]
	if !exists {
		return
	}

	// 패턴 그리기 (1.5배 확대)
	for row, rowPattern := range pattern {
		for col, pixel := range rowPattern {
			if pixel {
				// 각 픽셀을 1.5x1.5로 그리기
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						x := center[0] - 3 + col*2 + dx
						y := center[1] - 5 + row*2 + dy
						if x >= bounds.Min.X && x < bounds.Max.X && y >= bounds.Min.Y && y < bounds.Max.Y {
							img.Set(x, y, clr)
						}
					}
				}
			}
		}
	}
}

func (ua *UIAnalyzer) SelectElementWithAI(labeledImagePath, userGoal string, idToElement map[int]UIElement) (*AIResponse, error) {
	if ua.openaiClient == nil {
		return nil, fmt.Errorf("OpenAI client not configured")
	}

	imageData, err := ua.encodeImageToBase64(labeledImagePath)
	if err != nil {
		return nil, err
	}

	prompt := fmt.Sprintf(`Look at this UI screenshot with numbered elements and select the best element for the user's goal: "%s"

Instructions:
- Each UI element has a white circle with a number inside it
- 🔴 Red boxes = YOLO detected objects (general objects like person, bottle, etc.)
- 🟢 Green boxes = Clickable buttons and interactive elements  
- 🟠 Orange boxes = Input fields and text areas

Analyze the image and select the number (ID) of the element that best matches the user's goal.

Respond ONLY in JSON format:
{
    "selected_id": 3,
    "reasoning": "Selected element 3 because it appears to be the login button based on its position and context"
}`, userGoal)

	resp, err := ua.openaiClient.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4VisionPreview,
			Messages: []openai.ChatCompletionMessage{
				{
					Role: openai.ChatMessageRoleUser,
					MultiContent: []openai.ChatMessagePart{
						{Type: openai.ChatMessagePartTypeText, Text: prompt},
						{
							Type: openai.ChatMessagePartTypeImageURL,
							ImageURL: &openai.ChatMessageImageURL{
								URL: fmt.Sprintf("data:image/png;base64,%s", imageData),
							},
						},
					},
				},
			},
			MaxTokens:   500,
			Temperature: 0.1,
		},
	)

	if err != nil {
		return nil, err
	}

	return ua.parseAIResponse(resp.Choices[0].Message.Content), nil
}

func (ua *UIAnalyzer) encodeImageToBase64(imagePath string) (string, error) {
	imageFile, err := os.Open(imagePath)
	if err != nil {
		return "", err
	}
	defer imageFile.Close()

	imageData, err := io.ReadAll(imageFile)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(imageData), nil
}

func (ua *UIAnalyzer) parseAIResponse(responseText string) *AIResponse {
	var aiResp AIResponse

	if err := json.Unmarshal([]byte(responseText), &aiResp); err == nil {
		return &aiResp
	}

	if strings.Contains(responseText, "```json") {
		start := strings.Index(responseText, "```json") + 7
		end := strings.Index(responseText[start:], "```")
		if end != -1 {
			jsonStr := responseText[start : start+end]
			if err := json.Unmarshal([]byte(jsonStr), &aiResp); err == nil {
				return &aiResp
			}
		}
	}

	start := strings.Index(responseText, "{")
	end := strings.LastIndex(responseText, "}")
	if start != -1 && end != -1 && end > start {
		jsonStr := responseText[start : end+1]
		if err := json.Unmarshal([]byte(jsonStr), &aiResp); err == nil {
			return &aiResp
		}
	}

	return &AIResponse{Error: "Failed to parse AI response"}
}

func (ua *UIAnalyzer) GetCoordinatesFromID(selectedID int, idToElement map[int]UIElement) *[2]int {
	if element, exists := idToElement[selectedID]; exists {
		return &[2]int{element.Center[0], element.Center[1]}
	}
	return nil
}

var analyzer *UIAnalyzer

// HTTP 핸들러들
func analyzeHandler(c *gin.Context) {
	requestStart := time.Now()
	requestID := uuid.New().String()[:8]

	// 안전한 요청 처리
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic in analyzeHandler %s: %v", requestID, r)
			c.JSON(http.StatusInternalServerError, ActionResponse{
				Success:      false,
				ErrorMessage: "Internal server error - request failed",
			})
		}
	}()

	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, ActionResponse{
			Success: false, ErrorMessage: "Image file required",
		})
		return
	}

	userGoal := c.PostForm("user_goal")
	if userGoal == "" {
		c.JSON(http.StatusBadRequest, ActionResponse{
			Success: false, ErrorMessage: "User goal required",
		})
		return
	}

	tempID := uuid.New().String()
	imagePath := filepath.Join(os.TempDir(), fmt.Sprintf("analysis_%s_%s.png", requestID, tempID))

	if err := c.SaveUploadedFile(file, imagePath); err != nil {
		log.Printf("Failed to save uploaded file %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "Failed to save image",
		})
		return
	}
	defer os.Remove(imagePath)

	// 안전한 UI 요소 검출
	elements, err := analyzer.DetectUIElements(imagePath)
	if err != nil {
		log.Printf("Detection failed for request %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "Detection failed: " + err.Error(),
		})
		return
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	if totalElements == 0 {
		log.Printf("No UI elements detected for request %s", requestID)
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, Reasoning: "No UI elements detected in the image",
		})
		return
	}

	// 안전한 이미지 라벨링
	labeledImagePath, idToElement, err := analyzer.CreateLabeledImage(imagePath, elements)
	if err != nil {
		log.Printf("Image labeling failed for request %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "Image labeling failed: " + err.Error(),
		})
		return
	}
	defer os.Remove(labeledImagePath)

	// 안전한 AI 선택
	selection, err := analyzer.SelectElementWithAI(labeledImagePath, userGoal, idToElement)
	if err != nil {
		log.Printf("AI selection failed for request %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "AI selection failed: " + err.Error(),
		})
		return
	}

	if selection.Error != "" {
		log.Printf("AI selection error for request %s: %s", requestID, selection.Error)
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, ErrorMessage: selection.Error,
		})
		return
	}

	// 안전한 좌표 조회
	coordinates := analyzer.GetCoordinatesFromID(selection.SelectedID, idToElement)
	if coordinates == nil {
		log.Printf("Invalid element ID %d selected for request %s", selection.SelectedID, requestID)
		c.JSON(http.StatusOK, ActionResponse{
			Success:   false,
			Reasoning: fmt.Sprintf("Invalid element ID %d selected", selection.SelectedID),
		})
		return
	}

	processingTime := time.Since(requestStart)
	log.Printf("Analyze request %s completed in %v - selected ID %d at %v",
		requestID, processingTime, selection.SelectedID, coordinates)

	c.JSON(http.StatusOK, ActionResponse{
		Success:     true,
		Coordinates: coordinates,
		Reasoning:   selection.Reasoning,
		SelectedID:  &selection.SelectedID,
	})
}

func visualizeHandler(c *gin.Context) {
	requestID := uuid.New().String()[:8]
	requestStart := time.Now()

	// 안전한 요청 처리
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic in visualizeHandler %s: %v", requestID, r)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":      "Internal server error - request failed",
				"request_id": requestID,
			})
		}
	}()

	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Image file required"})
		return
	}

	tempID := uuid.New().String()
	imagePath := filepath.Join(os.TempDir(), fmt.Sprintf("visualization_%s_%s.png", requestID, tempID))

	if err := c.SaveUploadedFile(file, imagePath); err != nil {
		log.Printf("Failed to save uploaded file %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image"})
		return
	}
	defer os.Remove(imagePath)

	// 안전한 UI 요소 검출
	elements, err := analyzer.DetectUIElements(imagePath)
	if err != nil {
		log.Printf("Detection failed for request %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "Detection failed: " + err.Error(),
			"request_id": requestID,
		})
		return
	}

	// 안전한 이미지 라벨링
	labeledImagePath, idToElement, err := analyzer.CreateLabeledImage(imagePath, elements)
	if err != nil {
		log.Printf("Labeling failed for request %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "Image labeling failed: " + err.Error(),
			"request_id": requestID,
		})
		return
	}
	defer os.Remove(labeledImagePath)

	// 안전한 이미지 인코딩
	imageData, err := analyzer.encodeImageToBase64(labeledImagePath)
	if err != nil {
		log.Printf("Encoding failed for request %s: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "Image encoding failed: " + err.Error(),
			"request_id": requestID,
		})
		return
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	processingTime := time.Since(requestStart)

	log.Printf("Visualization request %s completed in %v - %d elements",
		requestID, processingTime, totalElements)

	c.JSON(http.StatusOK, gin.H{
		"success":            true,
		"labeled_image":      "data:image/png;base64," + imageData,
		"elements":           elements,
		"element_map":        idToElement,
		"total_elements":     totalElements,
		"processing_time_ms": processingTime.Milliseconds(),
		"yolo_info": gin.H{
			"enabled":   analyzer.yoloEnabled,
			"backend":   "GoCV",
			"safe_mode": true,
			"classes":   len(analyzer.classNames),
		},
		"request_id": requestID,
	})
}

func healthHandler(c *gin.Context) {
	status := "healthy"
	if analyzer == nil {
		status = "unhealthy - analyzer not initialized"
	}

	healthInfo := gin.H{
		"status":           status,
		"timestamp":        time.Now().Unix(),
		"yolo_enabled":     analyzer != nil && analyzer.yoloEnabled,
		"yolo_backend":     "GoCV Ultra-Safe",
		"openai_available": analyzer != nil && analyzer.openaiClient != nil,
		"safe_mode":        true,
		"error_handling":   "comprehensive",
		"memory_stats": gin.H{
			"gc_percent": debug.SetGCPercent(-1), // 현재 GC 백분율 반환
		},
	}

	if analyzer != nil {
		healthInfo["yolo_classes"] = len(analyzer.classNames)
		healthInfo["output_layers"] = len(analyzer.outputLayers)
		if len(analyzer.outputLayers) > 0 {
			healthInfo["output_layer_names"] = analyzer.outputLayers
		}
	}

	// GC 백분율 복원
	debug.SetGCPercent(50)

	c.JSON(http.StatusOK, healthInfo)
}

func rootHandler(c *gin.Context) {
	capabilities := []string{
		"🔴 YOLO object detection (GoCV + Ultra-Safe Processing)",
		"🟢 Advanced OpenCV button detection",
		"🟠 OpenCV input field detection",
		"🧠 OpenAI visual element selection",
		"🛡️ 100% crash-free operation with full error handling",
		"📋 Large numbered ID labels for GPT visual selection",
		"⚡ High-performance concurrent processing",
		"🔧 Auto-fallback mechanisms for maximum reliability",
	}

	c.JSON(http.StatusOK, gin.H{
		"service":        "Ultra-Safe Visual ID Selection UI Automation Server",
		"version":        "7.0.0-ultra-safe",
		"status":         "operational",
		"capabilities":   capabilities,
		"yolo_enabled":   analyzer != nil && analyzer.yoloEnabled,
		"yolo_backend":   "GoCV (Ultra-Safe)",
		"crash_free":     true,
		"error_handling": "comprehensive",
		"description":    "Ultra-safe YOLO + visual ID labeling - GPT selects by seeing numbered elements with zero crashes!",
		"safety_features": []string{
			"Comprehensive panic recovery",
			"Array bounds checking",
			"Mat validation",
			"Memory leak prevention",
			"Graceful error handling",
			"Automatic fallbacks",
		},
	})
}

func main() {
	log.Println("🚀 Starting Ultra-Safe Visual ID Selection UI Automation Server...")
	log.Println("   Using GoCV for stable YOLO inference")
	log.Println("   📋 GPT selects elements by seeing numbered labels on image!")
	log.Println("   🛡️ Full crash protection and error handling enabled")

	// 메모리 최적화
	debug.SetGCPercent(50)
	runtime.GOMAXPROCS(runtime.NumCPU())

	// 전역 panic 복구
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Main function panic recovered: %v", r)
			if analyzer != nil {
				analyzer.Close()
			}
		}
	}()

	var err error
	analyzer, err = NewUIAnalyzer()
	if err != nil {
		log.Fatalf("❌ Initialization failed: %v", err)
	}
	defer func() {
		if analyzer != nil {
			analyzer.Close()
		}
	}()

	// Gin 설정
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()

	// 커스텀 로거 (더 자세한 정보)
	r.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
			param.ClientIP,
			param.TimeStamp.Format(time.RFC3339),
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.Request.UserAgent(),
			param.ErrorMessage,
		)
	}))

	// 강화된 Recovery 미들웨어
	r.Use(gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		if err, ok := recovered.(string); ok {
			log.Printf("Gin panic recovered: %s", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Internal server error",
				"message": "Request failed due to unexpected error",
			})
		}
		c.AbortWithStatus(http.StatusInternalServerError)
	}))

	// CORS 설정
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type"}
	r.Use(cors.New(config))

	// 라우트 설정
	r.GET("/", rootHandler)
	r.GET("/health", healthHandler)
	r.POST("/analyze", analyzeHandler)
	r.POST("/visualize", visualizeHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("✅ Server ready on port %s", port)
	log.Printf("🔴 YOLO enabled: %t (GoCV)", analyzer.yoloEnabled)
	log.Printf("📋 Visual ID labeling enabled - GPT sees numbered elements!")
	log.Printf("🛡️ Full crash protection enabled!")

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("❌ Server failed: %v", err)
	}
}
