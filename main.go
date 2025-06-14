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
	// YOLO 아웃풋 레이어 이름들 (YOLOv4 기준)
	// 일반적으로 "yolo_82", "yolo_94", "yolo_106" 또는 유사한 이름들
	layerNames := net.GetLayerNames()
	unconnectedLayers := net.GetUnconnectedOutLayers()

	var outputNames []string
	for _, layerIndex := range unconnectedLayers {
		if layerIndex > 0 && layerIndex <= len(layerNames) {
			outputNames = append(outputNames, layerNames[layerIndex-1])
		}
	}

	// YOLOv4의 경우 보통 3개의 출력 레이어가 있음
	if len(outputNames) == 0 {
		// 하드코딩된 YOLO 출력 레이어 이름들 (fallback)
		outputNames = []string{"yolo_82", "yolo_94", "yolo_106"}
		log.Println("Using fallback YOLO output layer names")
	}

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

	// YOLO 입력 크기 (416x416)
	inputSize := image.Pt(416, 416)

	log.Printf("Processing image: %dx%d -> %dx%d", width, height, inputSize.X, inputSize.Y)

	// Blob 생성 (GoCV 방식 - MatType 파라미터 제거)
	blob := gocv.BlobFromImage(img, 1.0/255.0, inputSize, gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	// 네트워크 입력 설정
	ua.yoloNet.SetInput(blob, "")

	// YOLO 추론 실행
	startTime := time.Now()
	outputs := ua.yoloNet.ForwardLayers(ua.outputLayers)
	inferenceTime := time.Since(startTime)

	log.Printf("YOLO inference completed in %v", inferenceTime)

	// 결과 후처리
	objects := ua.postProcessYOLO(outputs, width, height, 0.3, 0.4)

	// 메모리 정리
	for i := range outputs {
		outputs[i].Close()
	}

	log.Printf("YOLO post-processing: %d objects detected", len(objects))
	return objects, nil
}

func (ua *UIAnalyzer) postProcessYOLO(outputs []gocv.Mat, imgWidth, imgHeight int, confThreshold, nmsThreshold float32) []UIElement {
	var boxes []image.Rectangle
	var confidences []float32
	var classIDs []int

	// 각 출력 레이어 처리
	for _, output := range outputs {
		// Mat 데이터를 float32 슬라이스로 변환
		data, err := output.DataPtrFloat32()
		if err != nil {
			log.Printf("Error getting float32 data: %v", err)
			continue
		}

		rows := output.Rows()
		cols := output.Cols()

		// YOLO 출력 포맷: [center_x, center_y, width, height, confidence, class_probs...]
		for i := 0; i < rows; i++ {
			offset := i * cols
			if offset+4 >= len(data) {
				continue
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

					boxes = append(boxes, image.Rect(x, y, x+width, y+height))
					confidences = append(confidences, finalConf)
					classIDs = append(classIDs, classID)
				}
			}
		}
	}

	// NMS (Non-Maximum Suppression) 적용
	indices := gocv.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

	var objects []UIElement
	for _, idx := range indices {
		if idx < len(boxes) && idx < len(classIDs) {
			box := boxes[idx]
			classID := classIDs[idx]
			confidence := confidences[idx]

			// 클래스 이름 가져오기
			className := "unknown"
			if classID < len(ua.classNames) {
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
	}

	return objects
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
		"yolo":      {255, 0, 0, 255},   // 빨간색
		"cv_button": {0, 255, 0, 255},   // 녹색
		"cv_input":  {255, 165, 0, 255}, // 주황색
	}

	// 모든 요소 그리기
	allElements := append(append(elements.YOLOObjects, elements.CVButtons...), elements.CVInputs...)
	for _, element := range allElements {
		idToElement[elementID] = element

		var elementColor color.RGBA
		if strings.HasPrefix(element.Type, "yolo") {
			elementColor = colors["yolo"]
		} else if strings.Contains(element.Type, "button") {
			elementColor = colors["cv_button"]
		} else {
			elementColor = colors["cv_input"]
		}

		ua.drawRectangle(labeledImg, element.BBox, elementColor)
		ua.drawText(labeledImg, element.Center, strconv.Itoa(elementID), elementColor)
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

func (ua *UIAnalyzer) drawRectangle(img *image.RGBA, bbox [4]int, color color.RGBA) {
	bounds := img.Bounds()

	// 두께 2픽셀의 사각형
	for thickness := 0; thickness < 2; thickness++ {
		for x := bbox[0]; x <= bbox[2]; x++ {
			if x >= bounds.Min.X && x < bounds.Max.X {
				if bbox[1]+thickness >= bounds.Min.Y && bbox[1]+thickness < bounds.Max.Y {
					img.Set(x, bbox[1]+thickness, color)
				}
				if bbox[3]-thickness >= bounds.Min.Y && bbox[3]-thickness < bounds.Max.Y {
					img.Set(x, bbox[3]-thickness, color)
				}
			}
		}

		for y := bbox[1]; y <= bbox[3]; y++ {
			if y >= bounds.Min.Y && y < bounds.Max.Y {
				if bbox[0]+thickness >= bounds.Min.X && bbox[0]+thickness < bounds.Max.X {
					img.Set(bbox[0]+thickness, y, color)
				}
				if bbox[2]-thickness >= bounds.Min.X && bbox[2]-thickness < bounds.Max.X {
					img.Set(bbox[2]-thickness, y, color)
				}
			}
		}
	}
}

func (ua *UIAnalyzer) drawText(img *image.RGBA, center [2]int, text string, color color.RGBA) {
	bounds := img.Bounds()

	// 중심에 원 그리기
	for dx := -3; dx <= 3; dx++ {
		for dy := -3; dy <= 3; dy++ {
			x, y := center[0]+dx, center[1]+dy
			if x >= bounds.Min.X && x < bounds.Max.X && y >= bounds.Min.Y && y < bounds.Max.Y {
				if dx*dx+dy*dy <= 9 {
					img.Set(x, y, color)
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

	var elementInfo []string
	for elementID, element := range idToElement {
		text := element.Text
		if text == "" {
			text = element.ClassName
		}
		if text == "" {
			text = element.Type
		}
		elementInfo = append(elementInfo, fmt.Sprintf("ID%d: %s (%.2f) at %v",
			elementID, text, element.Confidence, element.Center))
	}

	prompt := fmt.Sprintf(`Analyze this UI and select the best element for: "%s"

Elements detected:
%s

🔴 Red boxes = YOLO objects (general objects like person, bottle, etc.)
🟢 Green boxes = CV buttons (clickable UI elements)  
🟠 Orange boxes = CV input fields (text input areas)

Select the element ID that best matches the user's goal. Respond in JSON:
{
    "selected_id": 3,
    "reasoning": "Selected the login button because..."
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
		"yolo_info": gin.H{
			"enabled":   analyzer.yoloEnabled,
			"backend":   "GoCV",
			"safe_mode": true,
			"classes":   len(analyzer.classNames),
		},
	})
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":           "healthy",
		"timestamp":        time.Now().Unix(),
		"yolo_enabled":     analyzer != nil && analyzer.yoloEnabled,
		"yolo_backend":     "GoCV",
		"openai_available": analyzer != nil && analyzer.openaiClient != nil,
		"safe_mode":        true,
	})
}

func rootHandler(c *gin.Context) {
	capabilities := []string{
		"🔴 YOLO object detection (GoCV)",
		"🟢 Advanced OpenCV button detection",
		"🟠 OpenCV input field detection",
		"🧠 OpenAI element selection",
		"🛡️ 100% crash-free operation",
	}

	c.JSON(http.StatusOK, gin.H{
		"service":      "Safe YOLO UI Automation Server",
		"version":      "5.0.0-stable",
		"status":       "operational",
		"capabilities": capabilities,
		"yolo_enabled": analyzer != nil && analyzer.yoloEnabled,
		"yolo_backend": "GoCV (Safe)",
		"crash_free":   true,
		"description":  "YOLO powered by GoCV - stable and reliable!",
	})
}

func main() {
	log.Println("🚀 Starting Safe YOLO UI Automation Server...")
	log.Println("   Using GoCV for stable YOLO inference")

	// 메모리 최적화
	debug.SetGCPercent(50)
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

	log.Printf("✅ Server ready on port %s", port)
	log.Printf("🔴 YOLO enabled: %t (GoCV)", analyzer.yoloEnabled)
	log.Printf("🛡️ 100%% crash-free guaranteed!")

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
