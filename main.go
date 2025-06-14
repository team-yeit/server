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
	openaiClient *openai.Client
	initialized  bool
	mu           sync.RWMutex
}

func NewUIAnalyzer() (*UIAnalyzer, error) {
	log.Println("Initializing stable OpenCV-only UI automation system...")
	startTime := time.Now()

	// 메모리 최적화
	debug.SetGCPercent(50)
	runtime.GOMAXPROCS(runtime.NumCPU())

	analyzer := &UIAnalyzer{
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

	analyzer.initialized = true
	log.Printf("UI analyzer initialized in %v - OpenCV mode only", time.Since(startTime))
	log.Println("YOLO disabled for stability - using advanced OpenCV detection")

	return analyzer, nil
}

func (ua *UIAnalyzer) DetectUIElements(imagePath string) (*UIElements, error) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	if !ua.initialized {
		return nil, fmt.Errorf("analyzer not initialized")
	}

	log.Printf("Starting OpenCV UI element detection: %s", imagePath)
	startTime := time.Now()

	elements := &UIElements{
		YOLOObjects: []UIElement{}, // YOLO 완전 비활성화
		CVButtons:   []UIElement{},
		CVInputs:    []UIElement{},
	}

	// OpenCV 감지만 사용 (병렬 실행)
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

	// 추가 UI 요소 감지 (클릭 가능한 영역)
	clickables, err := ua.detectClickableElements(imagePath)
	if err != nil {
		log.Printf("Clickable detection failed: %v", err)
	} else {
		// 클릭 가능한 요소들을 버튼으로 분류
		elements.CVButtons = append(elements.CVButtons, clickables...)
		log.Printf("CV detected %d additional clickable elements", len(clickables))
	}

	total := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	log.Printf("OpenCV detection completed in %v - total: %d elements", time.Since(startTime), total)

	return elements, nil
}

func (ua *UIAnalyzer) detectCVButtons(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	var buttons []UIElement

	// 방법 1: 적응형 임계값 + 윤곽선 검출
	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.AdaptiveThreshold(img, &thresh, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 2)

	contours := gocv.FindContours(thresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		// 버튼 크기 필터링 (더 관대한 범위)
		if area > 500 && area < 100000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			// 버튼 특성 검사
			if aspectRatio > 0.2 && aspectRatio < 15.0 && rect.Dx() > 30 && rect.Dy() > 15 {
				rectArea := float64(rect.Dx() * rect.Dy())
				if area/rectArea > 0.5 { // 더 관대한 충실도
					buttons = append(buttons, UIElement{
						Type:       "cv_button",
						Confidence: 0.7,
						BBox:       [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
						Center:     [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2},
						Width:      rect.Dx(),
						Height:     rect.Dy(),
					})
				}
			}
		}
	}

	// 방법 2: 모폴로지 연산으로 추가 버튼 검출
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer kernel.Close()

	closed := gocv.NewMat()
	defer closed.Close()
	gocv.MorphologyEx(thresh, &closed, gocv.MorphClose, kernel)

	contours2 := gocv.FindContours(closed, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours2.Close()

	for i := 0; i < contours2.Size(); i++ {
		contour := contours2.At(i)
		area := gocv.ContourArea(contour)

		if area > 800 && area < 50000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 0.3 && aspectRatio < 10.0 && rect.Dx() > 40 && rect.Dy() > 20 {
				// 중복 검사
				isDuplicate := false
				newCenter := [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2}

				for _, existing := range buttons {
					dx := abs(existing.Center[0] - newCenter[0])
					dy := abs(existing.Center[1] - newCenter[1])
					if dx < 20 && dy < 20 { // 20픽셀 이내면 중복
						isDuplicate = true
						break
					}
				}

				if !isDuplicate {
					buttons = append(buttons, UIElement{
						Type:       "cv_button_morph",
						Confidence: 0.6,
						BBox:       [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
						Center:     newCenter,
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

	var inputs []UIElement

	// 방법 1: 에지 검출 + 수평 구조 요소
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(img, &edges, 30, 100)

	// 수평 구조 요소 (입력 필드 감지용)
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

		if area > 500 && area < 50000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			// 입력 필드 특성: 긴 직사각형
			if aspectRatio > 2.0 && rect.Dx() > 80 && rect.Dy() > 15 && rect.Dy() < 80 {
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

	// 방법 2: 텍스트 영역 검출 (입력 필드 후보)
	textKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(15, 3))
	defer textKernel.Close()

	textAreas := gocv.NewMat()
	defer textAreas.Close()
	gocv.MorphologyEx(edges, &textAreas, gocv.MorphClose, textKernel)

	contours2 := gocv.FindContours(textAreas, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours2.Close()

	for i := 0; i < contours2.Size(); i++ {
		contour := contours2.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 && area < 30000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 3.0 && rect.Dx() > 100 && rect.Dy() > 20 && rect.Dy() < 60 {
				// 중복 검사
				isDuplicate := false
				newCenter := [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2}

				for _, existing := range inputs {
					dx := abs(existing.Center[0] - newCenter[0])
					dy := abs(existing.Center[1] - newCenter[1])
					if dx < 30 && dy < 15 {
						isDuplicate = true
						break
					}
				}

				if !isDuplicate {
					inputs = append(inputs, UIElement{
						Type:       "cv_input_text",
						Confidence: 0.7,
						BBox:       [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
						Center:     newCenter,
						Width:      rect.Dx(),
						Height:     rect.Dy(),
					})
				}
			}
		}
	}

	return inputs, nil
}

func (ua *UIAnalyzer) detectClickableElements(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	var clickables []UIElement

	// 그래디언트 기반 검출 (버튼 경계 감지)
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	defer gradX.Close()
	defer gradY.Close()

	gocv.Sobel(img, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(img, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)

	absGradX := gocv.NewMat()
	absGradY := gocv.NewMat()
	defer absGradX.Close()
	defer absGradY.Close()

	gocv.ConvertScaleAbs(gradX, &absGradX, 1, 0)
	gocv.ConvertScaleAbs(gradY, &absGradY, 1, 0)

	grad := gocv.NewMat()
	defer grad.Close()
	gocv.AddWeighted(absGradX, 0.5, absGradY, 0.5, 0, &grad)

	// 임계값 적용
	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.Threshold(grad, &thresh, 50, 255, gocv.ThresholdBinary)

	// 모폴로지 연산
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(5, 5))
	defer kernel.Close()

	cleaned := gocv.NewMat()
	defer cleaned.Close()
	gocv.MorphologyEx(thresh, &cleaned, gocv.MorphClose, kernel)

	contours := gocv.FindContours(cleaned, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		// 클릭 가능한 요소 크기
		if area > 1000 && area < 80000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			// 적절한 종횡비의 사각형 영역
			if aspectRatio > 0.2 && aspectRatio < 8.0 && rect.Dx() > 25 && rect.Dy() > 25 {
				clickables = append(clickables, UIElement{
					Type:       "cv_clickable",
					Confidence: 0.5,
					BBox:       [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
					Center:     [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2},
					Width:      rect.Dx(),
					Height:     rect.Dy(),
				})
			}
		}
	}

	return clickables, nil
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
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
		"cv_button":    {0, 255, 0, 255},   // 녹색
		"cv_input":     {255, 165, 0, 255}, // 주황색
		"cv_clickable": {0, 191, 255, 255}, // 하늘색
	}

	// 모든 요소 그리기 (YOLO 제외)
	allElements := append(elements.CVButtons, elements.CVInputs...)
	for _, element := range allElements {
		idToElement[elementID] = element

		var elementColor color.RGBA
		if strings.Contains(element.Type, "button") {
			elementColor = colors["cv_button"]
		} else if strings.Contains(element.Type, "input") {
			elementColor = colors["cv_input"]
		} else {
			elementColor = colors["cv_clickable"]
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

	// 두께 2픽셀의 사각형 그리기
	for thickness := 0; thickness < 2; thickness++ {
		// 수평선
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

		// 수직선
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

	// 중심에 작은 원 그리기 (텍스트 대신)
	for dx := -2; dx <= 2; dx++ {
		for dy := -2; dy <= 2; dy++ {
			x, y := center[0]+dx, center[1]+dy
			if x >= bounds.Min.X && x < bounds.Max.X && y >= bounds.Min.Y && y < bounds.Max.Y {
				if dx*dx+dy*dy <= 4 { // 원 모양
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
		elementInfo = append(elementInfo, fmt.Sprintf("ID%d: %s confidence=%.2f position=%v size=%dx%d",
			elementID, text, element.Confidence, element.Center, element.Width, element.Height))
	}

	prompt := fmt.Sprintf(`Analyze this UI screenshot and select the best element for the user's goal: "%s"

Available UI Elements (detected by OpenCV):
%s

Instructions:
- Choose the element ID that best matches the user's goal
- Consider element type, position, and size
- Buttons are good for clicking actions
- Input fields are good for text entry
- Respond in JSON format only

Example response:
{
    "selected_id": 3,
    "reasoning": "Selected the login button at position (320, 150) as it matches the user's goal to log in"
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

// 핸들러 함수들
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
			Success: false, Reasoning: "No UI elements detected in the image",
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
			Success: false, ErrorMessage: "AI selection failed: " + err.Error(),
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
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Detection failed: " + err.Error()})
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
		"detection_info": gin.H{
			"yolo_enabled": false,
			"opencv_only":  true,
			"stable_mode":  true,
		},
	})
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":           "healthy",
		"timestamp":        time.Now().Unix(),
		"yolo_enabled":     false,
		"opencv_enabled":   true,
		"openai_available": analyzer != nil && analyzer.openaiClient != nil,
		"stable_mode":      true,
	})
}

func rootHandler(c *gin.Context) {
	capabilities := []string{
		"Advanced OpenCV button detection",
		"OpenCV input field detection",
		"Clickable element detection",
		"OpenAI element selection",
	}

	c.JSON(http.StatusOK, gin.H{
		"service":      "Stable UI Automation Server",
		"version":      "3.0.0-stable",
		"status":       "operational",
		"capabilities": capabilities,
		"yolo_enabled": false,
		"opencv_only":  true,
		"stable_mode":  true,
		"description":  "YOLO disabled for maximum stability. Using advanced OpenCV detection only.",
	})
}

func main() {
	log.Println("Starting stable UI automation server (OpenCV-only mode)...")

	// 메모리 최적화
	debug.SetGCPercent(50)
	runtime.GOMAXPROCS(runtime.NumCPU())

	var err error
	analyzer, err = NewUIAnalyzer()
	if err != nil {
		log.Fatalf("Initialization failed: %v", err)
	}

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

	log.Printf("🚀 Stable server ready on port %s", port)
	log.Println("✅ YOLO disabled - no more segmentation faults!")
	log.Println("🔧 Using advanced OpenCV detection algorithms")

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
