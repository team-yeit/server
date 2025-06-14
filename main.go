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

	"github.com/LdDl/go-darknet"
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
	yoloDetector   *darknet.YOLONetwork
	openaiClient   *openai.Client
	uiClassMapping map[string]string
	yoloMutex      sync.Mutex // YOLO 전용 뮤텍스 (절대 동시 실행 방지)
	yoloEnabled    bool
	initialized    bool
}

func NewUIAnalyzer() (*UIAnalyzer, error) {
	log.Println("Initializing UI automation system with thread-safe YOLO implementation...")
	startTime := time.Now()

	// CGO 메모리 관리 최적화
	debug.SetGCPercent(20)        // 더 빈번한 GC
	debug.SetMemoryLimit(2 << 30) // 2GB 메모리 제한
	runtime.GOMAXPROCS(runtime.NumCPU())

	analyzer := &UIAnalyzer{
		uiClassMapping: map[string]string{
			"person": "icon", "book": "button", "laptop": "screen", "mouse": "button",
			"keyboard": "input", "cell phone": "device", "tv": "screen", "remote": "button",
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

	// YOLO 초기화 (격리된 환경에서)
	if err := analyzer.initializeYOLO(); err != nil {
		log.Printf("YOLO initialization failed: %v - continuing with OpenCV only", err)
	}

	analyzer.initialized = true
	log.Printf("UI analyzer initialized in %v - YOLO: %t", time.Since(startTime), analyzer.yoloEnabled)
	return analyzer, nil
}

func (ua *UIAnalyzer) initializeYOLO() error {
	// 파일 존재 확인
	configPath := "cfg/yolov4.cfg"
	weightsPath := "yolov4.weights"

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return fmt.Errorf("config file not found: %s", configPath)
	}
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		return fmt.Errorf("weights file not found: %s", weightsPath)
	}

	// 파일 크기 검증 (YOLOv4 weights는 약 245MB)
	if stat, err := os.Stat(weightsPath); err == nil {
		if stat.Size() < 200*1024*1024 { // 200MB 미만이면 손상된 파일
			return fmt.Errorf("weights file too small: %d bytes", stat.Size())
		}
	}

	// YOLO 네트워크 설정 (안전한 파라미터)
	yoloDetector := &darknet.YOLONetwork{
		GPUDeviceIndex:           -1, // CPU 모드로 시작 (안정성 우선)
		NetworkConfigurationFile: configPath,
		WeightsFile:              weightsPath,
		Threshold:                0.3, // 더 높은 임계값으로 false positive 감소
	}

	// 초기화 시도 (패닉 복구)
	var initErr error
	func() {
		defer func() {
			if r := recover(); r != nil {
				initErr = fmt.Errorf("YOLO initialization panic: %v", r)
			}
		}()

		// 메모리 정렬 보장
		runtime.GC()
		initErr = yoloDetector.Init()
	}()

	if initErr != nil {
		return initErr
	}

	ua.yoloDetector = yoloDetector
	ua.yoloEnabled = true
	log.Printf("YOLO v4 initialized successfully (CPU mode, threshold: %.2f)", yoloDetector.Threshold)
	return nil
}

func (ua *UIAnalyzer) Close() {
	ua.yoloMutex.Lock()
	defer ua.yoloMutex.Unlock()

	if ua.yoloDetector != nil {
		func() {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("YOLO cleanup panic recovered: %v", r)
				}
			}()
			ua.yoloDetector.Close()
		}()
		ua.yoloDetector = nil
		log.Println("YOLO resources cleaned up")
	}
}

func (ua *UIAnalyzer) DetectUIElements(imagePath string) (*UIElements, error) {
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

	// YOLO 감지 (완전히 격리된 실행)
	if ua.yoloEnabled {
		yoloObjects, err := ua.detectYOLOObjectsSafe(imagePath)
		if err != nil {
			log.Printf("YOLO detection failed: %v - disabling YOLO", err)
			ua.yoloEnabled = false // 실패시 비활성화
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
		}
	}()

	go func() {
		defer wg.Done()
		inputs, err := ua.detectCVInputs(imagePath)
		if err != nil {
			log.Printf("CV input detection failed: %v", err)
		} else {
			elements.CVInputs = inputs
		}
	}()

	wg.Wait()

	total := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	log.Printf("Detection completed in %v - total: %d elements", time.Since(startTime), total)

	return elements, nil
}

func (ua *UIAnalyzer) detectYOLOObjectsSafe(imagePath string) ([]UIElement, error) {
	// 크리티컬 섹션: YOLO는 절대 동시 실행 불가
	ua.yoloMutex.Lock()
	defer ua.yoloMutex.Unlock()

	if !ua.yoloEnabled || ua.yoloDetector == nil {
		return []UIElement{}, nil
	}

	// 메모리 정리
	runtime.GC()

	// 이미지 로드 및 검증
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %v", err)
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// 이미지 크기 제한 (메모리 보호)
	if width <= 0 || height <= 0 || width > 2048 || height > 2048 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", width, height)
	}

	log.Printf("Processing image: %s (%dx%d)", format, width, height)

	// Darknet 이미지 변환 (안전한 메모리 할당)
	var darknetImg *darknet.DarknetImage
	var conversionErr error

	func() {
		defer func() {
			if r := recover(); r != nil {
				conversionErr = fmt.Errorf("image conversion panic: %v", r)
			}
		}()
		darknetImg, conversionErr = darknet.Image2Float32(img)
	}()

	if conversionErr != nil {
		return nil, conversionErr
	}
	if darknetImg == nil {
		return nil, fmt.Errorf("darknet image conversion returned nil")
	}

	// 안전한 리소스 정리
	defer func() {
		if darknetImg != nil {
			func() {
				defer func() {
					if r := recover(); r != nil {
						log.Printf("Image cleanup panic recovered: %v", r)
					}
				}()
				darknetImg.Close()
			}()
		}
		runtime.GC() // 즉시 메모리 정리
	}()

	// YOLO 추론 실행 (패닉 보호)
	var detections *darknet.DetectionResult
	var detectErr error

	func() {
		defer func() {
			if r := recover(); r != nil {
				detectErr = fmt.Errorf("YOLO detection panic: %v", r)
				ua.yoloEnabled = false // 패닉 발생시 비활성화
			}
		}()

		// 추가 메모리 정렬 보장
		runtime.GC()
		runtime.Gosched() // 스케줄러에게 제어권 양보

		detections, detectErr = ua.yoloDetector.Detect(darknetImg)
	}()

	if detectErr != nil {
		return nil, detectErr
	}

	if detections == nil || detections.Detections == nil {
		return []UIElement{}, nil
	}

	// 결과 처리
	var objects []UIElement
	processedCount := 0

	for _, detection := range detections.Detections {
		if detection == nil {
			continue
		}

		// 최고 확률 클래스 찾기
		maxProb := float32(0)
		bestClassIdx := 0

		for i, prob := range detection.Probabilities {
			if prob > maxProb {
				maxProb = prob
				bestClassIdx = i
			}
		}

		// 낮은 확률 필터링
		if maxProb < 0.3 {
			continue
		}

		// 클래스명 결정
		var className string
		if bestClassIdx < len(detection.ClassNames) && detection.ClassNames[bestClassIdx] != "" {
			className = detection.ClassNames[bestClassIdx]
		} else {
			className = "unknown"
		}

		// UI 타입 매핑
		uiType, exists := ua.uiClassMapping[className]
		if !exists {
			uiType = "object"
		}

		// 바운딩 박스 검증
		bbox := detection.BoundingBox
		x1, y1 := bbox.StartPoint.X, bbox.StartPoint.Y
		x2, y2 := bbox.EndPoint.X, bbox.EndPoint.Y

		// 경계 검사 및 유효성 검증
		if x1 < 0 || y1 < 0 || x2 > width || y2 > height || x1 >= x2 || y1 >= y2 {
			continue
		}

		centerX, centerY := (x1+x2)/2, (y1+y2)/2
		objWidth, objHeight := x2-x1, y2-y1

		// 최소 크기 검증
		if objWidth < 10 || objHeight < 10 {
			continue
		}

		objects = append(objects, UIElement{
			Type:       fmt.Sprintf("yolo_%s", uiType),
			ClassName:  className,
			Confidence: float64(maxProb),
			BBox:       [4]int{x1, y1, x2, y2},
			Center:     [2]int{centerX, centerY},
			Width:      objWidth,
			Height:     objHeight,
		})

		processedCount++
	}

	log.Printf("YOLO processed %d detections -> %d valid objects", len(detections.Detections), len(objects))
	return objects, nil
}

func (ua *UIAnalyzer) detectCVButtons(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	// 적응형 임계값 적용
	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.AdaptiveThreshold(img, &thresh, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 2)

	// 윤곽선 검출
	contours := gocv.FindContours(thresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	var buttons []UIElement
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		// 크기 필터링
		if area > 1000 && area < 50000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			// 종횡비 및 충실도 검사
			if aspectRatio > 0.3 && aspectRatio < 8.0 {
				rectArea := float64(rect.Dx() * rect.Dy())
				if area/rectArea > 0.7 {
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
	}

	return buttons, nil
}

func (ua *UIAnalyzer) detectCVInputs(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	// 에지 검출
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(img, &edges, 30, 100)

	// 수평 구조 요소
	horizontalKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(40, 1))
	defer horizontalKernel.Close()

	// 수평선 검출
	detectHorizontal := gocv.NewMat()
	defer detectHorizontal.Close()
	gocv.MorphologyEx(edges, &detectHorizontal, gocv.MorphOpen, horizontalKernel)

	contours := gocv.FindContours(detectHorizontal, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	var inputs []UIElement
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 && area < 30000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			// 입력 필드 특성 검사 (긴 직사각형)
			if aspectRatio > 3.0 && rect.Dx() > 100 && rect.Dy() > 20 && rect.Dy() < 60 {
				inputs = append(inputs, UIElement{
					Type:       "cv_input_field",
					Confidence: 0.7,
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
		"yolo":      {255, 0, 0, 255},
		"cv_button": {0, 255, 0, 255},
		"cv_input":  {255, 165, 0, 255},
	}

	// 모든 요소 그리기
	allElements := append(append(elements.YOLOObjects, elements.CVButtons...), elements.CVInputs...)
	for _, element := range allElements {
		idToElement[elementID] = element

		var elementColor color.RGBA
		if strings.HasPrefix(element.Type, "yolo") {
			elementColor = colors["yolo"]
		} else if strings.HasPrefix(element.Type, "cv_button") {
			elementColor = colors["cv_button"]
		} else {
			elementColor = colors["cv_input"]
		}

		ua.drawRectangle(labeledImg, element.BBox, elementColor)
		ua.drawText(labeledImg, element.Center, strconv.Itoa(elementID), elementColor)
		elementID++
	}

	// 임시 파일 저장
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

func (ua *UIAnalyzer) drawRectangle(img *image.RGBA, bbox [4]int, color color.RGBA) {
	bounds := img.Bounds()

	// 수평선 그리기
	for x := bbox[0]; x <= bbox[2]; x++ {
		if x >= bounds.Min.X && x < bounds.Max.X {
			if bbox[1] >= bounds.Min.Y && bbox[1] < bounds.Max.Y {
				img.Set(x, bbox[1], color)
			}
			if bbox[3] >= bounds.Min.Y && bbox[3] < bounds.Max.Y {
				img.Set(x, bbox[3], color)
			}
		}
	}

	// 수직선 그리기
	for y := bbox[1]; y <= bbox[3]; y++ {
		if y >= bounds.Min.Y && y < bounds.Max.Y {
			if bbox[0] >= bounds.Min.X && bbox[0] < bounds.Max.X {
				img.Set(bbox[0], y, color)
			}
			if bbox[2] >= bounds.Min.X && bbox[2] < bounds.Max.X {
				img.Set(bbox[2], y, color)
			}
		}
	}
}

func (ua *UIAnalyzer) drawText(img *image.RGBA, center [2]int, text string, color color.RGBA) {
	bounds := img.Bounds()
	if center[0] >= bounds.Min.X && center[0] < bounds.Max.X &&
		center[1] >= bounds.Min.Y && center[1] < bounds.Max.Y {
		img.Set(center[0], center[1], color)
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

	var elementInfo []string
	for elementID, element := range idToElement {
		text := element.Text
		if text == "" {
			text = element.ClassName
		}
		if text == "" {
			text = element.Type
		}
		elementInfo = append(elementInfo, fmt.Sprintf("ID%d: %s confidence=%.2f position=%v",
			elementID, text, element.Confidence, element.Center))
	}

	prompt := fmt.Sprintf(`Analyze this UI and select the best element for: "%s"

Elements:
%s

Respond in JSON:
{
    "selected_id": 1,
    "reasoning": "explanation"
}`, userGoal, strings.Join(elementInfo, "\n"))

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
			MaxTokens: 500,
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

	// JSON 파싱 시도
	if err := json.Unmarshal([]byte(responseText), &aiResp); err == nil {
		return &aiResp
	}

	// 코드 블록에서 JSON 추출
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

	// 일반 JSON 추출
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

// 핸들러 함수들 (동일)
func analyzeHandler(c *gin.Context) {
	requestStart := time.Now()
	requestID := uuid.New().String()[:8]

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
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "Failed to save image",
		})
		return
	}
	defer os.Remove(imagePath)

	elements, err := analyzer.DetectUIElements(imagePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: err.Error(),
		})
		return
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	if totalElements == 0 {
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, Reasoning: "No UI elements detected",
		})
		return
	}

	labeledImagePath, idToElement, err := analyzer.CreateLabeledImage(imagePath, elements)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "Image labeling failed",
		})
		return
	}
	defer os.Remove(labeledImagePath)

	selection, err := analyzer.SelectElementWithAI(labeledImagePath, userGoal, idToElement)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, ErrorMessage: "AI selection failed",
		})
		return
	}

	if selection.Error != "" {
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, ErrorMessage: selection.Error,
		})
		return
	}

	coordinates := analyzer.GetCoordinatesFromID(selection.SelectedID, idToElement)
	if coordinates == nil {
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, Reasoning: "Invalid element ID selected",
		})
		return
	}

	log.Printf("Request %s completed in %v", requestID, time.Since(requestStart))

	c.JSON(http.StatusOK, ActionResponse{
		Success: true, Coordinates: coordinates, Reasoning: selection.Reasoning,
		SelectedID: &selection.SelectedID,
	})
}

func visualizeHandler(c *gin.Context) {
	requestID := uuid.New().String()[:8]

	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Image file required"})
		return
	}

	tempID := uuid.New().String()
	imagePath := filepath.Join(os.TempDir(), fmt.Sprintf("visualization_%s_%s.png", requestID, tempID))

	if err := c.SaveUploadedFile(file, imagePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image"})
		return
	}
	defer os.Remove(imagePath)

	elements, err := analyzer.DetectUIElements(imagePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Detection failed"})
		return
	}

	labeledImagePath, idToElement, err := analyzer.CreateLabeledImage(imagePath, elements)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Labeling failed"})
		return
	}
	defer os.Remove(labeledImagePath)

	imageData, err := analyzer.encodeImageToBase64(labeledImagePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Encoding failed"})
		return
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)

	c.JSON(http.StatusOK, gin.H{
		"success":        true,
		"labeled_image":  "data:image/png;base64," + imageData,
		"elements":       elements,
		"element_map":    idToElement,
		"total_elements": totalElements,
	})
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":           "healthy",
		"timestamp":        time.Now().Unix(),
		"yolo_enabled":     analyzer != nil && analyzer.yoloEnabled,
		"openai_available": analyzer != nil && analyzer.openaiClient != nil,
	})
}

func rootHandler(c *gin.Context) {
	capabilities := []string{"OpenCV image processing", "OpenAI element selection"}
	if analyzer != nil && analyzer.yoloEnabled {
		capabilities = append([]string{"YOLO object detection"}, capabilities...)
	}

	c.JSON(http.StatusOK, gin.H{
		"service":      "UI Automation Server",
		"version":      "2.0.0",
		"status":       "operational",
		"capabilities": capabilities,
		"yolo_enabled": analyzer != nil && analyzer.yoloEnabled,
		"thread_safe":  true,
	})
}

func main() {
	log.Println("Starting thread-safe UI automation server...")

	// 메모리 및 성능 최적화
	debug.SetGCPercent(20)
	runtime.GOMAXPROCS(runtime.NumCPU())

	var err error
	analyzer, err = NewUIAnalyzer()
	if err != nil {
		log.Fatalf("Initialization failed: %v", err)
	}
	defer analyzer.Close()

	// Gin 설정
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

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

	log.Printf("Server ready on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
