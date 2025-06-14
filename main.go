package main

import (
	"bufio"
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

// ==================== 구조체 정의 ====================

type UIElement struct {
	Type       string  `json:"type"`
	Text       string  `json:"text,omitempty"`
	ClassName  string  `json:"class_name,omitempty"`
	Confidence float64 `json:"confidence"`
	BBox       [4]int  `json:"bbox"`   // [x1, y1, x2, y2]
	Center     [2]int  `json:"center"` // [x, y]
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

// ==================== 유틸리티 함수들 ====================

func loadClassNames(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var classNames []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		className := strings.TrimSpace(scanner.Text())
		if className != "" {
			classNames = append(classNames, className)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return classNames, nil
}

// ==================== UIAnalyzer 클래스 ====================

type UIAnalyzer struct {
	yoloDetector   *darknet.YOLONetwork
	openaiClient   *openai.Client
	uiClassMapping map[string]string
	mu             sync.RWMutex
}

func NewUIAnalyzer() (*UIAnalyzer, error) {
	log.Println("🚀 UI 분석 시스템 초기화 중...")

	// YOLO 모델 초기화 (go-darknet 사용)
	classNames, err := loadClassNames("coco.names")
	if err != nil {
		return nil, fmt.Errorf("클래스명 로드 실패: %v", err)
	}

	yoloDetector := &darknet.YOLONetwork{
		GPUDeviceIndex:           0,
		NetworkConfigurationFile: "yolov4.cfg",
		WeightsFile:              "yolov4.weights",
		Threshold:                0.25,
		ClassNames:               classNames,
		Classes:                  len(classNames),
	}

	err = yoloDetector.Init()
	if err != nil {
		return nil, fmt.Errorf("YOLO 초기화 실패: %v", err)
	}
	log.Println("✅ YOLO 모델 로드 완료")

	// OpenAI 클라이언트 초기화
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
	}

	openaiClient := openai.NewClient(apiKey)
	log.Println("✅ OpenAI API 연결 완료")

	// UI 클래스 매핑
	uiClassMapping := map[string]string{
		"person":     "icon",
		"book":       "button",
		"laptop":     "screen",
		"mouse":      "button",
		"keyboard":   "input",
		"cell phone": "device",
		"tv":         "screen",
		"remote":     "button",
	}

	return &UIAnalyzer{
		yoloDetector:   yoloDetector,
		openaiClient:   openaiClient,
		uiClassMapping: uiClassMapping,
	}, nil
}

func (ua *UIAnalyzer) Close() {
	if ua.yoloDetector != nil {
		ua.yoloDetector.Close()
	}
}

// ==================== UI 요소 검출 ====================

func (ua *UIAnalyzer) DetectUIElements(imagePath string) (*UIElements, error) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	elements := &UIElements{
		YOLOObjects: []UIElement{},
		CVButtons:   []UIElement{},
		CVInputs:    []UIElement{},
	}

	// 병렬로 검출 실행
	var wg sync.WaitGroup
	var yoloErr, cvButtonErr, cvInputErr error

	// YOLO 객체 검출
	wg.Add(1)
	go func() {
		defer wg.Done()
		yoloObjects, err := ua.detectYOLOObjects(imagePath)
		if err != nil {
			yoloErr = err
			return
		}
		elements.YOLOObjects = yoloObjects
	}()

	// OpenCV 버튼 검출
	wg.Add(1)
	go func() {
		defer wg.Done()
		buttons, err := ua.detectCVButtons(imagePath)
		if err != nil {
			cvButtonErr = err
			return
		}
		elements.CVButtons = buttons
	}()

	// OpenCV 입력필드 검출
	wg.Add(1)
	go func() {
		defer wg.Done()
		inputs, err := ua.detectCVInputs(imagePath)
		if err != nil {
			cvInputErr = err
			return
		}
		elements.CVInputs = inputs
	}()

	wg.Wait()

	// 에러 체크
	if yoloErr != nil {
		log.Printf("YOLO 검출 오류: %v", yoloErr)
	}
	if cvButtonErr != nil {
		log.Printf("CV 버튼 검출 오류: %v", cvButtonErr)
	}
	if cvInputErr != nil {
		log.Printf("CV 입력 검출 오류: %v", cvInputErr)
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	log.Printf("🔍 검출된 UI 요소: %d개", totalElements)

	return elements, nil
}

func (ua *UIAnalyzer) detectYOLOObjects(imagePath string) ([]UIElement, error) {
	// 이미지 파일 열기
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("이미지 파일 열기 실패: %v", err)
	}
	defer file.Close()

	// 이미지 디코딩
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("이미지 디코딩 실패: %v", err)
	}

	// DarknetImage로 변환
	darknetImg, err := darknet.Image2Float32(img)
	if err != nil {
		return nil, fmt.Errorf("DarknetImage 변환 실패: %v", err)
	}
	defer darknetImg.Close()

	// YOLO 검출 실행
	detections, err := ua.yoloDetector.Detect(darknetImg)
	if err != nil {
		return nil, fmt.Errorf("YOLO 검출 실패: %v", err)
	}

	var objects []UIElement
	for _, detection := range detections.Detections {
		// 가장 높은 확률의 클래스 찾기
		maxProb := float32(0)
		bestClassIdx := 0
		for i, prob := range detection.Probabilities {
			if prob > maxProb {
				maxProb = prob
				bestClassIdx = i
			}
		}

		// 신뢰도 임계값 확인
		if maxProb < 0.25 {
			continue
		}

		// 클래스명 가져오기
		var className string
		if bestClassIdx < len(detection.ClassNames) {
			className = detection.ClassNames[bestClassIdx]
		} else {
			className = "unknown"
		}

		// UI 타입 매핑
		uiType, exists := ua.uiClassMapping[className]
		if !exists {
			uiType = "object"
		}

		// 바운딩박스 좌표 (StartPoint, EndPoint는 image.Point)
		x1 := detection.BoundingBox.StartPoint.X
		y1 := detection.BoundingBox.StartPoint.Y
		x2 := detection.BoundingBox.EndPoint.X
		y2 := detection.BoundingBox.EndPoint.Y

		centerX := (x1 + x2) / 2
		centerY := (y1 + y2) / 2
		width := x2 - x1
		height := y2 - y1

		objects = append(objects, UIElement{
			Type:       fmt.Sprintf("yolo_%s", uiType),
			ClassName:  className,
			Confidence: float64(maxProb),
			BBox:       [4]int{x1, y1, x2, y2},
			Center:     [2]int{centerX, centerY},
			Width:      width,
			Height:     height,
		})
	}

	return objects, nil
}

func (ua *UIAnalyzer) detectCVButtons(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("이미지 로드 실패: %s", imagePath)
	}
	defer img.Close()

	// 적응형 임계값
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

		if area > 1000 && area < 50000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 0.3 && aspectRatio < 8.0 {
				// 간단한 검증: 면적과 바운딩 박스 면적 비율
				rectArea := float64(rect.Dx() * rect.Dy())
				if area/rectArea > 0.7 { // 바운딩 박스의 70% 이상을 차지하는 경우
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
		contour.Close()
	}

	return buttons, nil
}

func (ua *UIAnalyzer) detectCVInputs(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("이미지 로드 실패: %s", imagePath)
	}
	defer img.Close()

	// 엣지 검출
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(img, &edges, 30, 100)

	// 수평 커널로 입력필드 감지
	horizontalKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(40, 1))
	defer horizontalKernel.Close()

	detectHorizontal := gocv.NewMat()
	defer detectHorizontal.Close()
	gocv.MorphologyEx(edges, &detectHorizontal, gocv.MorphOpen, horizontalKernel)

	// 윤곽선 검출
	contours := gocv.FindContours(detectHorizontal, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	var inputs []UIElement
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 && area < 30000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

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
		contour.Close()
	}

	return inputs, nil
}

// ==================== 라벨링된 이미지 생성 ====================

func (ua *UIAnalyzer) CreateLabeledImage(imagePath string, elements *UIElements) (string, map[int]UIElement, error) {
	log.Println("🏷️ 라벨링된 이미지 생성 중...")

	// 원본 이미지 로드
	file, err := os.Open(imagePath)
	if err != nil {
		return "", nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return "", nil, err
	}

	// 새 이미지 생성 (그리기용)
	bounds := img.Bounds()
	labeledImg := image.NewRGBA(bounds)
	draw.Draw(labeledImg, bounds, img, bounds.Min, draw.Src)

	// ID 매핑과 색상 설정
	idToElement := make(map[int]UIElement)
	elementID := 1

	colors := map[string]color.RGBA{
		"yolo":      {255, 0, 0, 255},   // 빨간색
		"cv_button": {0, 255, 0, 255},   // 초록색
		"cv_input":  {255, 165, 0, 255}, // 주황색
	}

	// 모든 요소에 대해 라벨링
	allElements := []UIElement{}
	allElements = append(allElements, elements.YOLOObjects...)
	allElements = append(allElements, elements.CVButtons...)
	allElements = append(allElements, elements.CVInputs...)

	for _, element := range allElements {
		idToElement[elementID] = element

		// 색상 선택
		var elementColor color.RGBA
		if strings.HasPrefix(element.Type, "yolo") {
			elementColor = colors["yolo"]
		} else if strings.HasPrefix(element.Type, "cv_button") {
			elementColor = colors["cv_button"]
		} else {
			elementColor = colors["cv_input"]
		}

		// 바운딩 박스 그리기 (간단한 구현)
		ua.drawRectangle(labeledImg, element.BBox, elementColor)
		ua.drawText(labeledImg, element.Center, strconv.Itoa(elementID), elementColor)

		elementID++
	}

	// 임시 파일로 저장
	tempFile, err := os.CreateTemp("", "labeled_*.png")
	if err != nil {
		return "", nil, err
	}
	defer tempFile.Close()

	err = png.Encode(tempFile, labeledImg)
	if err != nil {
		return "", nil, err
	}

	totalElements := len(allElements)
	log.Printf("✅ 라벨링 이미지 생성 완료: %d개 요소", totalElements)

	return tempFile.Name(), idToElement, nil
}

func (ua *UIAnalyzer) drawRectangle(img *image.RGBA, bbox [4]int, color color.RGBA) {
	// 간단한 직사각형 그리기 (실제로는 더 정교한 구현 필요)
	for x := bbox[0]; x <= bbox[2]; x++ {
		if x >= 0 && x < img.Bounds().Dx() {
			if bbox[1] >= 0 && bbox[1] < img.Bounds().Dy() {
				img.Set(x, bbox[1], color)
			}
			if bbox[3] >= 0 && bbox[3] < img.Bounds().Dy() {
				img.Set(x, bbox[3], color)
			}
		}
	}
	for y := bbox[1]; y <= bbox[3]; y++ {
		if y >= 0 && y < img.Bounds().Dy() {
			if bbox[0] >= 0 && bbox[0] < img.Bounds().Dx() {
				img.Set(bbox[0], y, color)
			}
			if bbox[2] >= 0 && bbox[2] < img.Bounds().Dx() {
				img.Set(bbox[2], y, color)
			}
		}
	}
}

func (ua *UIAnalyzer) drawText(img *image.RGBA, center [2]int, text string, color color.RGBA) {
	// 간단한 텍스트 표시 (실제로는 폰트 라이브러리 필요)
	// 여기서는 중심점에만 마커 표시
	if center[0] >= 0 && center[0] < img.Bounds().Dx() && center[1] >= 0 && center[1] < img.Bounds().Dy() {
		img.Set(center[0], center[1], color)
	}
}

// ==================== AI 요소 선택 ====================

func (ua *UIAnalyzer) SelectElementWithAI(labeledImagePath, userGoal string, idToElement map[int]UIElement) (*AIResponse, error) {
	if ua.openaiClient == nil {
		return nil, fmt.Errorf("OpenAI API가 설정되지 않음")
	}

	log.Printf("🤖 AI가 목표 '%s'에 맞는 요소 선택 중...", userGoal)

	// 라벨링된 이미지를 base64로 인코딩
	imageData, err := ua.encodeImageToBase64(labeledImagePath)
	if err != nil {
		return nil, err
	}

	// ID별 요소 정보 생성
	var elementInfo []string
	for elementID, element := range idToElement {
		text := element.Text
		if text == "" {
			text = element.ClassName
		}
		if text == "" {
			text = element.Type
		}
		elementInfo = append(elementInfo, fmt.Sprintf("ID%d: %s - '%s' at %v",
			elementID, element.Type, text, element.Center))
	}

	prompt := fmt.Sprintf(`
이 스마트폰 화면 이미지를 보고, 사용자의 목표를 달성하기 위해 클릭해야 할 요소의 ID를 하나만 선택해주세요.

사용자 목표: "%s"

이미지에 표시된 요소들:
%s

다음 JSON 형식으로만 응답해주세요:
{
    "selected_id": 5,
    "reasoning": "이메일 변경을 위해 먼저 로그인 버튼을 클릭해야 합니다"
}

중요한 규칙:
1. 반드시 이미지에 표시된 ID 중 하나만 선택하세요
2. 사용자의 목표를 달성하기 위한 가장 적절한 다음 단계를 선택하세요
3. 현재 화면 상황을 고려하여 논리적인 선택을 하세요
4. 반드시 JSON 형식으로만 응답하세요`, userGoal, strings.Join(elementInfo, "\n"))

	resp, err := ua.openaiClient.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4VisionPreview,
			Messages: []openai.ChatCompletionMessage{
				{
					Role: openai.ChatMessageRoleUser,
					MultiContent: []openai.ChatMessagePart{
						{
							Type: openai.ChatMessagePartTypeText,
							Text: prompt,
						},
						{
							Type: openai.ChatMessagePartTypeImageURL,
							ImageURL: &openai.ChatMessageImageURL{
								URL:    fmt.Sprintf("data:image/png;base64,%s", imageData),
								Detail: openai.ImageURLDetailHigh,
							},
						},
					},
				},
			},
			MaxTokens:   1000,
			Temperature: 0.1,
		},
	)

	if err != nil {
		return nil, fmt.Errorf("OpenAI API 호출 실패: %v", err)
	}

	aiResponse := ua.parseAIResponse(resp.Choices[0].Message.Content)
	log.Printf("✅ AI 선택 완료: ID%d", aiResponse.SelectedID)

	return aiResponse, nil
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
	err := json.Unmarshal([]byte(responseText), &aiResp)
	if err == nil {
		return &aiResp
	}

	// JSON 블록 추출 시도
	if strings.Contains(responseText, "```json") {
		start := strings.Index(responseText, "```json") + 7
		end := strings.Index(responseText[start:], "```")
		if end != -1 {
			jsonStr := responseText[start : start+end]
			err = json.Unmarshal([]byte(jsonStr), &aiResp)
			if err == nil {
				return &aiResp
			}
		}
	}

	// 중괄호 블록 추출 시도
	start := strings.Index(responseText, "{")
	end := strings.LastIndex(responseText, "}")
	if start != -1 && end != -1 && end > start {
		jsonStr := responseText[start : end+1]
		err = json.Unmarshal([]byte(jsonStr), &aiResp)
		if err == nil {
			return &aiResp
		}
	}

	return &AIResponse{
		Error: fmt.Sprintf("JSON 파싱 실패: %s", responseText),
	}
}

func (ua *UIAnalyzer) GetCoordinatesFromID(selectedID int, idToElement map[int]UIElement) *[2]int {
	if element, exists := idToElement[selectedID]; exists {
		return &[2]int{element.Center[0], element.Center[1]}
	}
	return nil
}

// ==================== Gin 핸들러들 ====================

var analyzer *UIAnalyzer

func analyzeHandler(c *gin.Context) {
	// 이미지 파일 업로드 처리
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, ActionResponse{
			Success:      false,
			Reasoning:    "이미지 파일이 필요합니다",
			ErrorMessage: err.Error(),
		})
		return
	}

	// 사용자 목표
	userGoal := c.PostForm("user_goal")
	if userGoal == "" {
		c.JSON(http.StatusBadRequest, ActionResponse{
			Success:      false,
			Reasoning:    "user_goal이 필요합니다",
			ErrorMessage: "user_goal 파라미터가 없습니다",
		})
		return
	}

	// 임시 파일 저장
	tempID := uuid.New().String()
	originalImagePath := filepath.Join(os.TempDir(), fmt.Sprintf("original_%s.png", tempID))

	err = c.SaveUploadedFile(file, originalImagePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success:      false,
			Reasoning:    "이미지 저장 실패",
			ErrorMessage: err.Error(),
		})
		return
	}
	defer os.Remove(originalImagePath)

	log.Printf("📸 이미지 분석 시작: %s", userGoal)

	// 1. UI 요소 검출
	elements, err := analyzer.DetectUIElements(originalImagePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success:      false,
			Reasoning:    "UI 요소 검출 실패",
			ErrorMessage: err.Error(),
		})
		return
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	if totalElements == 0 {
		c.JSON(http.StatusOK, ActionResponse{
			Success:   false,
			Reasoning: "화면에서 클릭 가능한 UI 요소를 찾을 수 없습니다",
		})
		return
	}

	// 2. 라벨링된 이미지 생성
	labeledImagePath, idToElement, err := analyzer.CreateLabeledImage(originalImagePath, elements)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success:      false,
			Reasoning:    "라벨링된 이미지 생성 실패",
			ErrorMessage: err.Error(),
		})
		return
	}
	defer os.Remove(labeledImagePath)

	// 3. AI가 요소 선택
	selection, err := analyzer.SelectElementWithAI(labeledImagePath, userGoal, idToElement)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success:      false,
			Reasoning:    "AI 분석 실패",
			ErrorMessage: err.Error(),
		})
		return
	}

	if selection.Error != "" {
		c.JSON(http.StatusOK, ActionResponse{
			Success:      false,
			Reasoning:    selection.Error,
			ErrorMessage: selection.Error,
		})
		return
	}

	// 4. 좌표 추출
	coordinates := analyzer.GetCoordinatesFromID(selection.SelectedID, idToElement)
	if coordinates == nil {
		c.JSON(http.StatusOK, ActionResponse{
			Success:   false,
			Reasoning: fmt.Sprintf("선택된 ID%d의 좌표를 찾을 수 없습니다", selection.SelectedID),
		})
		return
	}

	log.Printf("✅ 분석 완료 - 선택된 ID: %d, 좌표: %v", selection.SelectedID, *coordinates)

	c.JSON(http.StatusOK, ActionResponse{
		Success:     true,
		Coordinates: coordinates,
		Reasoning:   selection.Reasoning,
		SelectedID:  &selection.SelectedID,
	})
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
	})
}

func rootHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message":          "지능형 UI 자동화 서버",
		"status":           "running",
		"openai_available": analyzer.openaiClient != nil,
	})
}

// ==================== 메인 함수 ====================

func main() {
	log.Println("🚀 지능형 UI 자동화 서버 시작")

	// 분석 시스템 초기화
	var err error
	analyzer, err = NewUIAnalyzer()
	if err != nil {
		log.Fatalf("❌ 서버 초기화 실패: %v", err)
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
	config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type", "Authorization"}
	r.Use(cors.New(config))

	// 라우트 설정
	r.GET("/", rootHandler)
	r.GET("/health", healthHandler)
	r.POST("/analyze", analyzeHandler)

	// 서버 시작
	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("📖 API 문서: http://localhost:%s", port)
	log.Printf("✅ 서버 시작: 포트 %s", port)

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("❌ 서버 시작 실패: %v", err)
	}
}
