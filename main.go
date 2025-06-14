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
	return classNames, scanner.Err()
}

type UIAnalyzer struct {
	yoloDetector   *darknet.YOLONetwork
	openaiClient   *openai.Client
	uiClassMapping map[string]string
	mu             sync.RWMutex
}

func NewUIAnalyzer() (*UIAnalyzer, error) {
	log.Println("Initializing intelligent UI automation system with YOLO object detection and OpenAI integration...")
	startTime := time.Now()

	classNames, err := loadClassNames("coco.names")
	if err != nil {
		return nil, fmt.Errorf("failed to load COCO class names from file: %v", err)
	}
	log.Printf("Successfully loaded %d object classes from COCO dataset for YOLO detection", len(classNames))

	yoloDetector := &darknet.YOLONetwork{
		GPUDeviceIndex:           0,
		NetworkConfigurationFile: "yolov4.cfg",
		WeightsFile:              "yolov4.weights",
		Threshold:                0.25,
		ClassNames:               classNames,
		Classes:                  len(classNames),
	}

	if err := yoloDetector.Init(); err != nil {
		return nil, fmt.Errorf("YOLO network initialization failed - check model files and configuration: %v", err)
	}
	log.Printf("YOLO v4 neural network successfully initialized with confidence threshold %.2f and %d detection classes", yoloDetector.Threshold, yoloDetector.Classes)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("WARNING: OPENAI_API_KEY environment variable not set - AI element selection will be disabled")
	} else {
		log.Println("OpenAI API client configured successfully for intelligent element selection and reasoning")
	}

	openaiClient := openai.NewClient(apiKey)
	uiClassMapping := map[string]string{
		"person": "icon", "book": "button", "laptop": "screen", "mouse": "button",
		"keyboard": "input", "cell phone": "device", "tv": "screen", "remote": "button",
	}

	initDuration := time.Since(startTime)
	log.Printf("UI analysis system fully initialized in %v with YOLO detection, OpenCV processing, and AI selection capabilities", initDuration)

	return &UIAnalyzer{
		yoloDetector: yoloDetector, openaiClient: openaiClient, uiClassMapping: uiClassMapping,
	}, nil
}

func (ua *UIAnalyzer) Close() {
	if ua.yoloDetector != nil {
		ua.yoloDetector.Close()
		log.Println("YOLO detector resources cleaned up and released")
	}
}

func (ua *UIAnalyzer) DetectUIElements(imagePath string) (*UIElements, error) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	log.Printf("Starting comprehensive UI element detection on image: %s", imagePath)
	startTime := time.Now()

	elements := &UIElements{YOLOObjects: []UIElement{}, CVButtons: []UIElement{}, CVInputs: []UIElement{}}
	var wg sync.WaitGroup
	var yoloErr, cvButtonErr, cvInputErr error

	wg.Add(3)
	go func() {
		defer wg.Done()
		log.Println("Executing YOLO object detection for UI elements recognition...")
		yoloObjects, err := ua.detectYOLOObjects(imagePath)
		if err != nil {
			yoloErr = err
			log.Printf("YOLO detection failed with error: %v", err)
			return
		}
		elements.YOLOObjects = yoloObjects
		log.Printf("YOLO detection completed successfully - found %d potential UI objects", len(yoloObjects))
	}()

	go func() {
		defer wg.Done()
		log.Println("Executing OpenCV button detection using contour analysis and morphological operations...")
		buttons, err := ua.detectCVButtons(imagePath)
		if err != nil {
			cvButtonErr = err
			log.Printf("OpenCV button detection failed with error: %v", err)
			return
		}
		elements.CVButtons = buttons
		log.Printf("OpenCV button detection completed - identified %d button-like elements", len(buttons))
	}()

	go func() {
		defer wg.Done()
		log.Println("Executing OpenCV input field detection using edge detection and horizontal kernel filtering...")
		inputs, err := ua.detectCVInputs(imagePath)
		if err != nil {
			cvInputErr = err
			log.Printf("OpenCV input field detection failed with error: %v", err)
			return
		}
		elements.CVInputs = inputs
		log.Printf("OpenCV input field detection completed - discovered %d input field candidates", len(inputs))
	}()

	wg.Wait()

	if yoloErr != nil || cvButtonErr != nil || cvInputErr != nil {
		log.Printf("Detection errors encountered - YOLO: %v, CV Buttons: %v, CV Inputs: %v", yoloErr, cvButtonErr, cvInputErr)
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	detectionDuration := time.Since(startTime)
	log.Printf("Comprehensive UI element detection completed in %v - total elements detected: %d (YOLO: %d, Buttons: %d, Inputs: %d)",
		detectionDuration, totalElements, len(elements.YOLOObjects), len(elements.CVButtons), len(elements.CVInputs))

	return elements, nil
}

func (ua *UIAnalyzer) detectYOLOObjects(imagePath string) ([]UIElement, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file for YOLO processing: %v", err)
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("image decoding failed - unsupported format or corrupted file: %v", err)
	}
	log.Printf("Image successfully decoded - format: %s, dimensions: %dx%d", format, img.Bounds().Dx(), img.Bounds().Dy())

	darknetImg, err := darknet.Image2Float32(img)
	if err != nil {
		return nil, fmt.Errorf("darknet image conversion failed - memory allocation or format issue: %v", err)
	}
	defer darknetImg.Close()

	detections, err := ua.yoloDetector.Detect(darknetImg)
	if err != nil {
		return nil, fmt.Errorf("YOLO neural network inference failed: %v", err)
	}

	var objects []UIElement
	detectionCount := 0
	filteredCount := 0

	for _, detection := range detections.Detections {
		detectionCount++
		maxProb := float32(0)
		bestClassIdx := 0
		for i, prob := range detection.Probabilities {
			if prob > maxProb {
				maxProb = prob
				bestClassIdx = i
			}
		}

		if maxProb < 0.25 {
			filteredCount++
			continue
		}

		var className string
		if bestClassIdx < len(detection.ClassNames) {
			className = detection.ClassNames[bestClassIdx]
		} else {
			className = "unknown"
		}

		uiType, exists := ua.uiClassMapping[className]
		if !exists {
			uiType = "object"
		}

		x1, y1 := detection.BoundingBox.StartPoint.X, detection.BoundingBox.StartPoint.Y
		x2, y2 := detection.BoundingBox.EndPoint.X, detection.BoundingBox.EndPoint.Y
		centerX, centerY := (x1+x2)/2, (y1+y2)/2
		width, height := x2-x1, y2-y1

		objects = append(objects, UIElement{
			Type: fmt.Sprintf("yolo_%s", uiType), ClassName: className, Confidence: float64(maxProb),
			BBox: [4]int{x1, y1, x2, y2}, Center: [2]int{centerX, centerY}, Width: width, Height: height,
		})
	}

	log.Printf("YOLO processing results - raw detections: %d, confidence filtered: %d, final objects: %d",
		detectionCount, filteredCount, len(objects))
	return objects, nil
}

func (ua *UIAnalyzer) detectCVButtons(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("OpenCV failed to load image for button detection: %s", imagePath)
	}
	defer img.Close()

	log.Printf("Processing image for button detection - applying adaptive thresholding and contour analysis...")
	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.AdaptiveThreshold(img, &thresh, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 2)

	contours := gocv.FindContours(thresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	var buttons []UIElement
	totalContours := contours.Size()
	validContours := 0

	for i := 0; i < totalContours; i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 && area < 50000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 0.3 && aspectRatio < 8.0 {
				rectArea := float64(rect.Dx() * rect.Dy())
				if area/rectArea > 0.7 {
					buttons = append(buttons, UIElement{
						Type: "cv_button", Confidence: 0.8,
						BBox:   [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
						Center: [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2},
						Width:  rect.Dx(), Height: rect.Dy(),
					})
					validContours++
				}
			}
		}
		contour.Close()
	}

	log.Printf("Button detection analysis - total contours: %d, aspect ratio filtered: %d, final buttons: %d",
		totalContours, validContours, len(buttons))
	return buttons, nil
}

func (ua *UIAnalyzer) detectCVInputs(imagePath string) ([]UIElement, error) {
	img := gocv.IMRead(imagePath, gocv.IMReadGrayScale)
	if img.Empty() {
		return nil, fmt.Errorf("OpenCV failed to load image for input field detection: %s", imagePath)
	}
	defer img.Close()

	log.Printf("Processing image for input field detection - applying Canny edge detection and morphological filtering...")
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(img, &edges, 30, 100)

	horizontalKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(40, 1))
	defer horizontalKernel.Close()

	detectHorizontal := gocv.NewMat()
	defer detectHorizontal.Close()
	gocv.MorphologyEx(edges, &detectHorizontal, gocv.MorphOpen, horizontalKernel)

	contours := gocv.FindContours(detectHorizontal, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	var inputs []UIElement
	totalContours := contours.Size()
	validInputs := 0

	for i := 0; i < totalContours; i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 && area < 30000 {
			rect := gocv.BoundingRect(contour)
			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			if aspectRatio > 3.0 && rect.Dx() > 100 && rect.Dy() > 20 && rect.Dy() < 60 {
				inputs = append(inputs, UIElement{
					Type: "cv_input_field", Confidence: 0.7,
					BBox:   [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
					Center: [2]int{rect.Min.X + rect.Dx()/2, rect.Min.Y + rect.Dy()/2},
					Width:  rect.Dx(), Height: rect.Dy(),
				})
				validInputs++
			}
		}
		contour.Close()
	}

	log.Printf("Input field detection analysis - total contours: %d, dimension filtered: %d, final input fields: %d",
		totalContours, validInputs, len(inputs))
	return inputs, nil
}

func (ua *UIAnalyzer) CreateLabeledImage(imagePath string, elements *UIElements) (string, map[int]UIElement, error) {
	log.Printf("Generating labeled visualization image with element IDs and bounding boxes...")
	startTime := time.Now()

	file, err := os.Open(imagePath)
	if err != nil {
		return "", nil, fmt.Errorf("failed to open original image for labeling: %v", err)
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return "", nil, fmt.Errorf("image decoding failed during label generation: %v", err)
	}
	log.Printf("Original image loaded for labeling - format: %s, size: %dx%d", format, img.Bounds().Dx(), img.Bounds().Dy())

	bounds := img.Bounds()
	labeledImg := image.NewRGBA(bounds)
	draw.Draw(labeledImg, bounds, img, bounds.Min, draw.Src)

	idToElement := make(map[int]UIElement)
	elementID := 1

	colors := map[string]color.RGBA{
		"yolo": {255, 0, 0, 255}, "cv_button": {0, 255, 0, 255}, "cv_input": {255, 165, 0, 255},
	}

	allElements := append(append(elements.YOLOObjects, elements.CVButtons...), elements.CVInputs...)
	drawnElements := 0

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
		drawnElements++
		elementID++
	}

	tempFile, err := os.CreateTemp("", "labeled_ui_*.png")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create temporary file for labeled image: %v", err)
	}
	defer tempFile.Close()

	if err := png.Encode(tempFile, labeledImg); err != nil {
		return "", nil, fmt.Errorf("PNG encoding failed for labeled image: %v", err)
	}

	labelingDuration := time.Since(startTime)
	log.Printf("Labeled image generation completed in %v - elements drawn: %d, output file: %s",
		labelingDuration, drawnElements, tempFile.Name())

	return tempFile.Name(), idToElement, nil
}

func (ua *UIAnalyzer) drawRectangle(img *image.RGBA, bbox [4]int, color color.RGBA) {
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
	if center[0] >= 0 && center[0] < img.Bounds().Dx() && center[1] >= 0 && center[1] < img.Bounds().Dy() {
		img.Set(center[0], center[1], color)
	}
}

func (ua *UIAnalyzer) SelectElementWithAI(labeledImagePath, userGoal string, idToElement map[int]UIElement) (*AIResponse, error) {
	if ua.openaiClient == nil {
		return nil, fmt.Errorf("OpenAI API client not configured - check OPENAI_API_KEY environment variable")
	}

	log.Printf("Initiating AI-powered element selection for user goal: '%s' with %d available elements", userGoal, len(idToElement))
	startTime := time.Now()

	imageData, err := ua.encodeImageToBase64(labeledImagePath)
	if err != nil {
		return nil, fmt.Errorf("base64 image encoding failed for AI analysis: %v", err)
	}
	log.Printf("Image successfully encoded to base64 for OpenAI vision API - size: %d bytes", len(imageData))

	var elementInfo []string
	for elementID, element := range idToElement {
		text := element.Text
		if text == "" {
			text = element.ClassName
		}
		if text == "" {
			text = element.Type
		}
		elementInfo = append(elementInfo, fmt.Sprintf("ID%d: %s type='%s' confidence=%.2f position=%v size=%dx%d",
			elementID, element.Type, text, element.Confidence, element.Center, element.Width, element.Height))
	}

	prompt := fmt.Sprintf(`Analyze this smartphone screen interface and select the most appropriate UI element to achieve the user's goal.

User Goal: "%s"

Available Interactive Elements:
%s

Respond ONLY in this exact JSON format:
{
    "selected_id": 5,
    "reasoning": "Detailed explanation of why this element was selected based on the user's goal and current screen context"
}

Selection Criteria:
1. Choose exactly ONE element ID from the list above
2. Select the element that best progresses toward the user's stated goal
3. Consider the logical sequence of actions required to complete the task
4. Analyze the current screen state and available options
5. Provide detailed reasoning explaining your selection strategy`, userGoal, strings.Join(elementInfo, "\n"))

	log.Printf("Sending vision analysis request to OpenAI GPT-4V with prompt length: %d characters", len(prompt))

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
								URL: fmt.Sprintf("data:image/png;base64,%s", imageData), Detail: openai.ImageURLDetailHigh,
							},
						},
					},
				},
			},
			MaxTokens: 1000, Temperature: 0.1,
		},
	)

	if err != nil {
		return nil, fmt.Errorf("OpenAI API request failed - check API key and quota: %v", err)
	}

	aiResponse := ua.parseAIResponse(resp.Choices[0].Message.Content)
	aiDuration := time.Since(startTime)

	if aiResponse.Error != "" {
		log.Printf("AI element selection failed in %v - parsing error: %s", aiDuration, aiResponse.Error)
	} else {
		log.Printf("AI element selection completed successfully in %v - selected ID: %d, reasoning: %s",
			aiDuration, aiResponse.SelectedID, aiResponse.Reasoning)
	}

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

	return &AIResponse{Error: fmt.Sprintf("JSON parsing failed for AI response: %s", responseText)}
}

func (ua *UIAnalyzer) GetCoordinatesFromID(selectedID int, idToElement map[int]UIElement) *[2]int {
	if element, exists := idToElement[selectedID]; exists {
		return &[2]int{element.Center[0], element.Center[1]}
	}
	return nil
}

var analyzer *UIAnalyzer

func analyzeHandler(c *gin.Context) {
	requestStart := time.Now()
	requestID := uuid.New().String()[:8]

	file, err := c.FormFile("image")
	if err != nil {
		log.Printf("Request %s failed - missing image file parameter: %v", requestID, err)
		c.JSON(http.StatusBadRequest, ActionResponse{
			Success: false, Reasoning: "Image file parameter required for analysis",
			ErrorMessage: err.Error(),
		})
		return
	}

	userGoal := c.PostForm("user_goal")
	if userGoal == "" {
		log.Printf("Request %s failed - missing user_goal parameter", requestID)
		c.JSON(http.StatusBadRequest, ActionResponse{
			Success: false, Reasoning: "User goal parameter required for AI selection",
			ErrorMessage: "user_goal parameter missing from request",
		})
		return
	}

	log.Printf("Processing analysis request %s - goal: '%s', image: %s (%.2f KB)",
		requestID, userGoal, file.Filename, float64(file.Size)/1024)

	tempID := uuid.New().String()
	originalImagePath := filepath.Join(os.TempDir(), fmt.Sprintf("analysis_%s_%s.png", requestID, tempID))

	if err := c.SaveUploadedFile(file, originalImagePath); err != nil {
		log.Printf("Request %s failed - image save error: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, Reasoning: "Failed to save uploaded image for processing",
			ErrorMessage: err.Error(),
		})
		return
	}
	defer os.Remove(originalImagePath)

	elements, err := analyzer.DetectUIElements(originalImagePath)
	if err != nil {
		log.Printf("Request %s failed - UI detection error: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, Reasoning: "UI element detection pipeline failed",
			ErrorMessage: err.Error(),
		})
		return
	}

	totalElements := len(elements.YOLOObjects) + len(elements.CVButtons) + len(elements.CVInputs)
	if totalElements == 0 {
		log.Printf("Request %s completed - no interactive elements detected in image", requestID)
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, Reasoning: "No clickable UI elements detected in the provided screenshot",
		})
		return
	}

	labeledImagePath, idToElement, err := analyzer.CreateLabeledImage(originalImagePath, elements)
	if err != nil {
		log.Printf("Request %s failed - image labeling error: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, Reasoning: "Failed to generate labeled image for AI analysis",
			ErrorMessage: err.Error(),
		})
		return
	}
	defer os.Remove(labeledImagePath)

	selection, err := analyzer.SelectElementWithAI(labeledImagePath, userGoal, idToElement)
	if err != nil {
		log.Printf("Request %s failed - AI selection error: %v", requestID, err)
		c.JSON(http.StatusInternalServerError, ActionResponse{
			Success: false, Reasoning: "AI element selection process failed",
			ErrorMessage: err.Error(),
		})
		return
	}

	if selection.Error != "" {
		log.Printf("Request %s failed - AI response parsing error: %s", requestID, selection.Error)
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, Reasoning: selection.Error, ErrorMessage: selection.Error,
		})
		return
	}

	coordinates := analyzer.GetCoordinatesFromID(selection.SelectedID, idToElement)
	if coordinates == nil {
		log.Printf("Request %s failed - invalid element ID selected: %d", requestID, selection.SelectedID)
		c.JSON(http.StatusOK, ActionResponse{
			Success: false, Reasoning: fmt.Sprintf("Selected element ID %d not found in detected elements", selection.SelectedID),
		})
		return
	}

	requestDuration := time.Since(requestStart)
	selectedElement := idToElement[selection.SelectedID]
	log.Printf("Request %s completed successfully in %v - selected element ID %d (%s) at coordinates %v with confidence %.2f",
		requestID, requestDuration, selection.SelectedID, selectedElement.Type, *coordinates, selectedElement.Confidence)

	c.JSON(http.StatusOK, ActionResponse{
		Success: true, Coordinates: coordinates, Reasoning: selection.Reasoning, SelectedID: &selection.SelectedID,
	})
}

func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "healthy", "timestamp": time.Now().Unix(),
		"yolo_initialized":  analyzer != nil && analyzer.yoloDetector != nil,
		"openai_configured": analyzer != nil && analyzer.openaiClient != nil,
	})
}

func rootHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":          "Intelligent UI Automation Server",
		"version":          "1.0.0",
		"status":           "operational",
		"capabilities":     []string{"YOLO object detection", "OpenCV image processing", "OpenAI element selection"},
		"openai_available": analyzer.openaiClient != nil,
	})
}

func main() {
	log.Println("Starting Intelligent UI Automation Server with advanced computer vision and AI capabilities...")
	serverStart := time.Now()

	var err error
	analyzer, err = NewUIAnalyzer()
	if err != nil {
		log.Fatalf("Critical initialization failure - server cannot start: %v", err)
	}
	defer analyzer.Close()

	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type", "Authorization"}
	r.Use(cors.New(config))

	r.GET("/", rootHandler)
	r.GET("/health", healthHandler)
	r.POST("/analyze", analyzeHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	initDuration := time.Since(serverStart)
	log.Printf("Server initialization completed in %v - all systems operational", initDuration)
	log.Printf("API endpoints available:")
	log.Printf("  GET  / - Service information and capabilities")
	log.Printf("  GET  /health - Health check and system status")
	log.Printf("  POST /analyze - UI element analysis and AI selection")
	log.Printf("Server listening on port %s - ready to process UI automation requests", port)

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Server startup failed on port %s: %v", port, err)
	}
}
