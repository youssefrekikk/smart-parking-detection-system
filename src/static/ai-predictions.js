/**
 * AI Predictions Module
 * Handles auto-generation of parking spot annotations using the trained model
 */

class AIPredictions {
    constructor(canvas, fabricObjects) {
        this.canvas = canvas;
        this.fabricObjects = fabricObjects;
        this.isGenerating = false;
        this.confidenceThreshold = 0.3;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Auto-generate button
        const autoGenerateBtn = document.getElementById('auto-generate-boxes');
        if (autoGenerateBtn) {
            autoGenerateBtn.addEventListener('click', () => this.autoGenerateBoxes());
        }

        // Confidence threshold slider
        const confidenceSlider = document.getElementById('confidence-threshold');
        const confidenceValue = document.getElementById('confidence-value');
        if (confidenceSlider && confidenceValue) {
            confidenceSlider.addEventListener('input', (e) => {
                this.confidenceThreshold = parseFloat(e.target.value);
                confidenceValue.textContent = this.confidenceThreshold;
            });
        }
    }

    /**
     * Auto-generate parking spot boxes using AI model
     */
    async autoGenerateBoxes() {
        if (this.isGenerating) {
            console.log('Already generating boxes...');
            return;
        }

        const button = document.getElementById('auto-generate-boxes');
        const loadingOverlay = document.getElementById('loading-overlay');
        
        try {
            this.isGenerating = true;
            this.showLoadingState(button, loadingOverlay);
            
            // Get current frame from video feed
            const frameBlob = await this.captureCurrentFrame();
            
            // Send to backend for AI processing
            const result = await this.sendFrameForProcessing(frameBlob);
            
            // Process the results
            await this.processAIResults(result);
            
        } catch (error) {
            console.error('Error generating boxes:', error);
            this.showError(error.message);
        } finally {
            this.isGenerating = false;
            this.hideLoadingState(button, loadingOverlay);
        }
    }

    /**
     * Capture current frame from video feed
     */
    async captureCurrentFrame() {
        const videoElement = document.getElementById('video-feed');
        
        if (!videoElement || !videoElement.src) {
            throw new Error('No video feed available. Please start a camera feed first.');
        }

        // Create a canvas to capture the current frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Set canvas dimensions to match video
        tempCanvas.width = videoElement.naturalWidth || videoElement.width || 640;
        tempCanvas.height = videoElement.naturalHeight || videoElement.height || 480;
        
        // Draw the current video frame
        tempCtx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
        
        // Convert to blob
        return new Promise((resolve, reject) => {
            tempCanvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to capture frame'));
                }
            }, 'image/jpeg', 0.8);
        });
    }

    /**
     * Send frame to backend for AI processing
     */
    async sendFrameForProcessing(frameBlob) {
        const formData = new FormData();
        formData.append('image', frameBlob, 'frame.jpg');
        formData.append('confidence', this.confidenceThreshold.toString());
        
        const response = await fetch('/api/auto-generate-boxes', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Process AI results and add boxes to canvas
     */
    async processAIResults(result) {
        if (!result.boxes || result.boxes.length === 0) {
            this.showNoDetectionsMessage();
            return;
        }

        // Ask user if they want to clear existing boxes
        const shouldClear = await this.confirmClearExistingBoxes(result.boxes.length);
        
        if (shouldClear) {
            this.clearAllBoxes();
        }
        
        // Add generated boxes to canvas
        this.addGeneratedBoxesToCanvas(result.boxes);
        
        // Show success message
        this.showSuccessMessage(result.boxes.length);
        
        console.log(`Generated ${result.boxes.length} parking spot boxes`);
    }

    /**
     * Add generated boxes to canvas
     */
    addGeneratedBoxesToCanvas(boxes) {
        if (!this.canvas) {
            console.error('Canvas not initialized');
            return;
        }
        
        const canvasWidth = this.canvas.getWidth();
        const canvasHeight = this.canvas.getHeight();
        
        boxes.forEach((box, index) => {
            const rect = this.createFabricRectFromBox(box, index, canvasWidth, canvasHeight);
            
            // Add to canvas and tracking
            this.canvas.add(rect);
            this.fabricObjects.push(rect);
        });
        
        // Update the box list and render
        if (window.updateBoxList) {
            window.updateBoxList();
        }
        this.canvas.renderAll();
    }

    /**
     * Create a Fabric.js rectangle from AI detection box
     */
    createFabricRectFromBox(box, index, canvasWidth, canvasHeight) {
        // Convert normalized coordinates to canvas coordinates
        const left = box.x1 * canvasWidth;
        const top = box.y1 * canvasHeight;
        const width = (box.x2 - box.x1) * canvasWidth;
        const height = (box.y2 - box.y1) * canvasHeight;
        
        // Create fabric rectangle
        const rect = new fabric.Rect({
            left: left,
            top: top,
            width: width,
            height: height,
            fill: 'transparent',
            stroke: this.getGroupColor(box.group_id || 'default'),
            strokeWidth: 2,
            selectable: true,
            hasControls: true,
            hasBorders: true,
            angle: box.angle || 0
        });
        
        // Add metadata
        rect.boxId = `ai_generated_${Date.now()}_${index}`;
        rect.groupId = box.group_id || 'default';
        rect.confidence = box.confidence || 0;
        rect.isAIGenerated = true;
        
        return rect;
    }

    /**
     * Get color for group
     */
    getGroupColor(groupId) {
        const colors = {
            'group_1': '#FF6B6B',  // Red
            'group_2': '#4ECDC4',  // Teal  
            'group_3': '#45B7D1',  // Blue
            'group_4': '#96CEB4',  // Green
            'group_5': '#FFEAA7',  // Yellow
            'group_6': '#DDA0DD',  // Plum
            'default': '#FF9F43'   // Orange
        };
        return colors[groupId] || colors['default'];
    }

    /**
     * Clear all boxes from canvas
     */
    clearAllBoxes() {
        if (!this.canvas) return;
        
        // Remove all objects from canvas
        this.canvas.getObjects().forEach(obj => {
            this.canvas.remove(obj);
        });
        
        // Clear tracking arrays
        this.fabricObjects.length = 0;
        
        // Update UI
        if (window.updateBoxList) {
            window.updateBoxList();
        }
        this.canvas.renderAll();
    }

    /**
     * Show loading state
     */
    showLoadingState(button, loadingOverlay) {
        if (button) {
            button.disabled = true;
            button.innerHTML = '⏳ Generating...';
        }
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }
    }

    /**
     * Hide loading state
     */
    hideLoadingState(button, loadingOverlay) {
        if (button) {
            button.disabled = false;
            button.innerHTML = '🤖 Auto-Generate Boxes';
        }
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        alert(`Error generating boxes: ${message}`);
    }

    /**
     * Show success message
     */
    showSuccessMessage(count) {
        // Create a temporary success notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;
        notification.textContent = `✅ Successfully generated ${count} parking spot${count > 1 ? 's' : ''}!`;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    /**
     * Show message when no detections found
     */
    showNoDetectionsMessage() {
        alert('No parking spots detected. Try:\n' +
              '• Adjusting the confidence threshold\n' +
              '• Ensuring the image contains parking areas\n' +
              '• Checking if the camera feed is clear');
    }

    /**
     * Confirm if user wants to clear existing boxes
     */
    async confirmClearExistingBoxes(newBoxCount) {
        if (this.fabricObjects.length === 0) {
            return true; // No existing boxes, no need to ask
        }

        return confirm(
            `Found ${newBoxCount} parking spots!\n\n` +
            `You currently have ${this.fabricObjects.length} existing box${this.fabricObjects.length > 1 ? 'es' : ''}.\n` +
            `Do you want to clear existing boxes and replace them with AI-generated ones?\n\n` +
            `Click "OK" to replace, or "Cancel" to add to existing boxes.`
        );
    }

    /**
     * Update confidence threshold
     */
    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0.1, Math.min(0.9, threshold));
        
        const slider = document.getElementById('confidence-threshold');
        const value = document.getElementById('confidence-value');
        
        if (slider) slider.value = this.confidenceThreshold;
        if (value) value.textContent = this.confidenceThreshold;
    }

    /**
     * Get current confidence threshold
     */
    getConfidenceThreshold() {
        return this.confidenceThreshold;
    }

    /**
     * Check if currently generating
     */
    isCurrentlyGenerating() {
        return this.isGenerating;
    }
}

// Export for use in other modules
window.AIPredictions = AIPredictions;
