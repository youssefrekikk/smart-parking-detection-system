// Fabric.js canvas
let canvas;
let fabricInitialized = false;
let isDrawing = false;
let startPoint;
let currentRect;
let video = document.getElementById('video-feed');
let fabricObjects = [];
let selectedColor = 'green';
let isDrawMode = false;
let isResizing = false;
let isMoving = false;
let isMouseDown = false; // Track mouse state for drawing

let aiPredictions;

// Initialize Fabric.js canvas
function initFabric() {
    console.log("Initializing Fabric.js canvas");
    if (fabricInitialized) {
        // Clean up properly before reinitializing
        adjustCanvasSize();
        return;
    }

    const canvasContainer = document.getElementById('canvas-container');
    const drawingCanvas = document.getElementById('drawing-canvas');
    
    // Create Fabric canvas
    canvas = new fabric.Canvas('drawing-canvas', {
        selection: true,
        preserveObjectStacking: true
    });
    
    fabricInitialized = true;
    
    // Set event listeners
    canvas.on('mouse:down', handleMouseDown);
    canvas.on('mouse:move', handleMouseMove);
    canvas.on('mouse:up', handleMouseUp);
    
    // Add mouseout handler to ensure drawing completes properly
    canvas.on('mouse:out', function(o) {
        if (isDrawMode && currentRect && isMouseDown) {
            // Continue drawing even when mouse is outside canvas
            const pointer = canvas.getPointer(o.e);
            continueDrawingRect({ e: o.e });  // Pass the event to continue drawing
        }
    });
    
    // Document-wide mouse move/up to handle cases when mouse leaves canvas
    document.addEventListener('mousemove', function(e) {
        if (isDrawMode && currentRect && isMouseDown) {
            const canvasElement = canvas.upperCanvasEl;
            const rect = canvasElement.getBoundingClientRect();
            
            // Convert page coordinates to canvas coordinates
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Only process if coordinates are reasonable
            if (x >= 0 && y >= 0) {
                const fakeEvent = { e: { clientX: e.clientX, clientY: e.clientY } };
                continueDrawingRect(fakeEvent);
            }
        }
    });
    
    document.addEventListener('mouseup', function(e) {
        if (isDrawMode && currentRect && isMouseDown) {
            isMouseDown = false;
            
            // Only finalize if we have a reasonable size
            if (currentRect.width > 10 && currentRect.height > 10) {
                // Add to our tracked objects
                fabricObjects.push(currentRect);
                addBoxToList(currentRect);
                currentRect = null;
                startPoint = null;
                canvas.renderAll();
            } else if (currentRect) {
                canvas.remove(currentRect);
                currentRect = null;
                startPoint = null;
                canvas.renderAll();
            }
        }
    });
    
    // Object modification events
    canvas.on('object:modified', function(e) {
        console.log("Object modified");
        updateBoxList();
    });
    
    // Track when we're resizing or moving to prevent creating new boxes
    canvas.on('object:scaling:before', function(e) {
        console.log("Scaling started");
        isResizing = true;
    });
    
    canvas.on('object:moving:before', function(e) {
        console.log("Moving started");
        isMoving = true;
    });
    
    // After scaling or moving is done
    canvas.on('object:modified', function(e) {
        console.log("Object modified - resetting flags");
        isResizing = false;
        isMoving = false;
    });
    
    canvas.on('mouse:up', function() {
        console.log("Mouse up - resetting flags");
        isResizing = false;
        isMoving = false;
    });
    
    canvas.on('selection:created', function(e) {
        console.log("Selection created", e.selected);
        if (e.selected && e.selected.length > 0) {
            updateSelectedBox(e.selected[0]);
        }
    });
    
    // Make canvas responsive
    window.addEventListener('resize', adjustCanvasSize);
    
    // Set initial mode
    setDrawingMode(false);
    
    // Initial canvas sizing
    adjustCanvasSize();
    window.canvas = canvas;
    window.fabricObjects = fabricObjects;
    // Initialize grouping functionality
    if (typeof window.initGrouping === 'function') {
        console.log("Initializing grouping functionality");
        window.initGrouping(canvas, fabricObjects);
    } else {
        console.error("Grouping functionality not available");
    }
}

function adjustCanvasSize() {
    if (!fabricInitialized) return;
    
    // Get the video container and image dimensions
    const videoContainer = document.getElementById('video-container');
    const containerRect = videoContainer.getBoundingClientRect();
    const imageRect = video.getBoundingClientRect();
    
    // Get the canvas container and set its size to match the image
    const canvasContainer = document.getElementById('canvas-container');
    canvasContainer.style.width = containerRect.width + 'px';
    canvasContainer.style.height = containerRect.height + 'px';
    
    // Make canvas container unselectable
    canvasContainer.style.userSelect = 'none';
    canvasContainer.style.webkitUserSelect = 'none';
    canvasContainer.style.msUserSelect = 'none';
    
    // Position canvas container to overlay perfectly on the video container
    canvasContainer.style.top = '0';
    canvasContainer.style.left = '0';
    
    // Important: Set canvas dimensions to match the actual element size, not just CSS
    // This ensures the drawing area extends to the full container
    canvas.setWidth(containerRect.width);
    canvas.setHeight(containerRect.height);
    
    // If video has loaded, calculate the scaling factor to map coordinates correctly
    if (video.naturalWidth && video.naturalHeight) {
        // Calculate the scale to fit the image in the container
        const imageAspectRatio = video.naturalWidth / video.naturalHeight;
        const containerAspectRatio = containerRect.width / containerRect.height;
        
        // Calculate displayed image dimensions within the container
        let displayedWidth, displayedHeight;
        if (imageAspectRatio > containerAspectRatio) {
            // Image is wider than container
            displayedWidth = containerRect.width;
            displayedHeight = containerRect.width / imageAspectRatio;
        } else {
            // Image is taller than container
            displayedHeight = containerRect.height;
            displayedWidth = containerRect.height * imageAspectRatio;
        }
        
        // Calculate offsets for centering the image
        const offsetX = (containerRect.width - displayedWidth) / 2;
        const offsetY = (containerRect.height - displayedHeight) / 2;
        
        // Store these values as data attributes for coordinate mapping
        canvasContainer.dataset.offsetX = offsetX;
        canvasContainer.dataset.offsetY = offsetY;
        canvasContainer.dataset.scale = displayedWidth / video.naturalWidth;
    }
    
    // Enable interaction with the canvas
    canvasContainer.style.pointerEvents = 'auto';
    
    // Render the canvas
    canvas.renderAll();
}

// Handle mouse events based on current mode
function handleMouseDown(o) {
    isMouseDown = true;
    // Handle group selection if in group mode and we have a target
    if (window.isGroupSelectionMode && o.target) {
        return window.handleObjectSelection(o.target, canvas);
    }
    // Drawing mode handling
    if (isDrawMode && !isResizing && !isMoving && !o.target) {
        startDrawingRect(o);
    }
}

function handleMouseMove(o) {
    if (isDrawMode && currentRect && isMouseDown && !isResizing && !isMoving) {
        continueDrawingRect(o);
    }
}

function handleMouseUp(o) {
    isMouseDown = false;
    if (isDrawMode && currentRect && !isResizing && !isMoving) {
        finishDrawingRect(o);
    }
}

// Set the drawing or editing mode
function setDrawingMode(enabled) {
    isDrawMode = enabled;
    
    // Disable selection and object manipulation while in drawing mode
    if (enabled) {
        canvas.selection = false;
        canvas.discardActiveObject();
        canvas.forEachObject(function(obj) {
            obj.selectable = false;
            obj.evented = false;
        });
    } else {
        canvas.selection = true;
        canvas.forEachObject(function(obj) {
            obj.selectable = true;
            obj.evented = true;
        });
    }
    
    // Ensure the canvas container has pointer events enabled based on the mode
    const canvasContainer = document.getElementById('canvas-container');
    canvasContainer.style.pointerEvents = 'auto';
    
    // Reset current drawing state if switching to edit mode
    if (!enabled && currentRect) {
        canvas.remove(currentRect);
        currentRect = null;
        startPoint = null;
    }
    
    // Reset flags
    isResizing = false;
    isMoving = false;
    
    canvas.renderAll();
}

// Drawing functions
function startDrawingRect(o) {
    const pointer = canvas.getPointer(o.e);
    startPoint = pointer;
    
    currentRect = new fabric.Rect({
        left: pointer.x,
        top: pointer.y,
        width: 0,
        height: 0,
        stroke: selectedColor,
        strokeWidth: 2,
        fill: 'transparent',
        transparentCorners: false,
        cornerColor: 'white',
        cornerStrokeColor: 'black',
        borderColor: 'black',
        cornerSize: 10,
        padding: 5,
        cornerStyle: 'circle',
        hasRotatingPoint: true,
        centeredRotation: true
    });
    
    canvas.add(currentRect);
    canvas.renderAll();
}

function continueDrawingRect(o) {
    if (!currentRect || !startPoint) return;
    
    const pointer = canvas.getPointer(o.e);
    
    if (startPoint.x > pointer.x) {
        currentRect.set({ left: pointer.x });
    }
    if (startPoint.y > pointer.y) {
        currentRect.set({ top: pointer.y });
    }
    
    currentRect.set({
        width: Math.abs(startPoint.x - pointer.x),
        height: Math.abs(startPoint.y - pointer.y)
    });
    
    canvas.renderAll();
}

function finishDrawingRect(o) {
    if (!currentRect) return;
    
    // Only add if size is reasonable
    if (currentRect.width > 10 && currentRect.height > 10) {
        // Add to our tracked objects
        fabricObjects.push(currentRect);
        addBoxToList(currentRect);
    } else {
        canvas.remove(currentRect);
    }
    
    currentRect = null;
    startPoint = null;
    canvas.renderAll();
}

function updateSelectedBox(fabricObject) {
    if (!fabricObject) return;
    
    // Update UI based on selected object
    const index = fabricObjects.indexOf(fabricObject);
    if (index !== -1) {
        const boxList = document.getElementById('box-list');
        const boxItems = boxList.getElementsByClassName('box-item');
        
        for (let i = 0; i < boxItems.length; i++) {
            boxItems[i].classList.remove('selected');
        }
        
        if (index < boxItems.length) {
            boxItems[index].classList.add('selected');
        }
    }
}

function addBoxToList(fabricObject) {
    const boxList = document.getElementById('box-list');
    const boxItem = document.createElement('div');
    boxItem.className = 'box-item';
    const boxIndex = fabricObjects.indexOf(fabricObject);
    
    boxItem.innerHTML = `
        <div class="angle-control">
            <span>Box ${boxIndex + 1}</span>
            <input type="number" class="box-angle" value="${fabricObject.angle || 0}" min="0" max="360" 
                   onchange="updateBoxAngle(${boxIndex}, this.value)">
        </div>
        <button onclick="removeBox(${boxIndex})">Delete</button>
    `;
    
    boxItem.addEventListener('click', function() {
        canvas.setActiveObject(fabricObjects[boxIndex]);
        canvas.renderAll();
        updateSelectedBox(fabricObjects[boxIndex]);
    });
    
    boxList.appendChild(boxItem);
}

function updateBoxList() {
    const boxList = document.getElementById('box-list');
    boxList.innerHTML = '';
    
    fabricObjects.forEach((obj, index) => {
        if (obj && obj.type === 'rect') {
            addBoxToList(obj);
        }
    });
}

function updateBoxAngle(index, angle) {
    if (index >= 0 && index < fabricObjects.length) {
        const obj = fabricObjects[index];
        obj.set('angle', parseFloat(angle));
        canvas.renderAll();
    }
}

function removeBox(index) {
    if (index >= 0 && index < fabricObjects.length) {
        canvas.remove(fabricObjects[index]);
        fabricObjects.splice(index, 1);
        updateBoxList();
        canvas.renderAll();
    }
}

// Capture frame from camera
function captureFrame(cameraId) {
    fetch(`http://localhost:8000/capture-frame/${cameraId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to capture frame');
            }
            return response.blob();
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            video.src = url;
            video.style.display = 'block';
            video.onload = () => {
                initFabric();
                adjustCanvasSize();
            };
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
        });
}

// Modified save mask function to include group information
// Modified save mask function to include group information
function saveMask() {
    const cameraId = document.getElementById('camera-id').value;
    if (!cameraId) {
        alert('Please register a camera first');
        return;
    }

    if (!fabricObjects || fabricObjects.length === 0) {
        alert('Please draw at least one box before saving');
        return;
    }

    try {
        // Create enhanced boxes with group information
        const enhancedBoxes = fabricObjects.map((obj, index) => {
            const boundingRect = obj.getBoundingRect();
            return {
                index: index,
                x1: boundingRect.left / canvas.width,
                y1: boundingRect.top / canvas.height,
                x2: (boundingRect.left + boundingRect.width) / canvas.width,
                y2: (boundingRect.top + boundingRect.height) / canvas.height,
                angle: obj.angle || 0,
                groupId: obj.groupId || null,
                location: obj.groupLocation || null
            };
        });

        // Create group information
        const groupInfo = [];
        if (window.boxGroups && window.boxGroups.length > 0) {
            window.boxGroups.forEach(group => {
                const boxIndices = group.boxes.map(box => fabricObjects.indexOf(box));
                groupInfo.push({
                    group_id: group.id,
                    location: group.location,
                    box_indices: boxIndices.filter(idx => idx !== -1) // Only include valid indices
                });
            });
        }

        // Create the enhanced data structure for both server and download
        const enhancedData = {
            camera_id: cameraId,
            boxes: enhancedBoxes,
            groups: groupInfo
        };

        console.log("Sending data to server:", JSON.stringify(enhancedData));

        // Save enhanced format to server
        fetch(`http://localhost:8000/save-mask/${cameraId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(enhancedData) // Send the enhanced format to the server
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.detail || 'Failed to save mask');
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Mask saved:', data);
            
            // Download the same enhanced data as JSON file
            const jsonString = JSON.stringify(enhancedData, null, 2);
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `mask_${cameraId}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('Mask saved successfully and downloaded as JSON!');
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`Failed to save mask: ${error.message}`);
        });
    } catch (error) {
        console.error('Error preparing data:', error);
        alert(`Failed to prepare mask data: ${error.message}`);
    }
}



// Initialize the event listeners once the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded, setting up event listeners");
    
    const autoGenerateBtn = document.getElementById('auto-generate-boxes');
    if (autoGenerateBtn) {
        console.log("Auto-generate button found, adding event listener");
        autoGenerateBtn.addEventListener('click', autoGenerateBoxes);
    } else {
        console.error("Auto-generate button not found!");
    }
    
    // Add confidence threshold slider handler - ADD THESE LINES
    const confidenceSlider = document.getElementById('confidence-threshold');
    const confidenceValue = document.getElementById('confidence-value');
    if (confidenceSlider && confidenceValue) {
        console.log("Confidence slider found, adding event listener");
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value;
        });
    } else {
        console.error("Confidence slider not found!");
    }
    // Register event listeners for UI controls
    document.getElementById('add-camera').addEventListener('click', () => {
        const cameraId = document.getElementById('camera-id').value;
        const cameraType = document.getElementById('camera-type').value;
        const cameraLocation = document.getElementById('camera-location').value;
        const cameraUrl = document.getElementById('camera-url').value;
        const videoFile = document.getElementById('video-file').files[0];

        if (!cameraId || !cameraLocation) {
            alert('Please fill in all required fields');
            return;
        }

        if (cameraType === 'ip_camera' && !cameraUrl) {
            alert('Please provide a camera URL for IP camera');
            return;
        }

        if (cameraType === 'video_file' && !videoFile) {
            alert('Please select a video file');
            return;
        }

        // Create FormData to handle file upload
        const formData = new FormData();
        formData.append('id', cameraId);
        formData.append('type', cameraType);
        formData.append('location', cameraLocation);
        
        if (cameraType === 'ip_camera') {
            formData.append('url', cameraUrl);
        }
        
        if (cameraType === 'video_file' && videoFile) {
            formData.append('video_file', videoFile);
        }

        fetch('http://localhost:8000/register-camera', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to register camera');
            }
            return response.json();
        })
        .then(data => {
            console.log('Camera registered:', data);
            captureFrame(cameraId);
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
        });
    });

    document.getElementById('start-drawing').addEventListener('click', function() {
        isDrawing = !isDrawing;
        this.textContent = isDrawing ? 'Stop Drawing' : 'Start Drawing';
        
        // Set drawing mode
        setDrawingMode(isDrawing);
        
        document.getElementById('angle-control').style.display = isDrawing ? 'flex' : 'none';
    });

    document.getElementById('clear-boxes').addEventListener('click', () => {
        console.log("Clear boxes clicked");
        canvas.clear();
        fabricObjects = [];
        document.getElementById('box-list').innerHTML = '';
    });

    document.getElementById('box-color').addEventListener('change', function() {
        selectedColor = this.value;
    });

    document.getElementById('save-mask').addEventListener('click', saveMask);

    // Fix for camera type selection - immediately show/hide appropriate fields
    const cameraTypeSelect = document.getElementById('camera-type');


    





    // Function to update input fields visibility based on camera type
    function updateCameraInputFields() {
        const type = cameraTypeSelect.value;
        const urlGroup = document.getElementById('url-input-group');
        const videoGroup = document.getElementById('video-file-group');
        
        if (type === 'ip_camera') {
            urlGroup.style.display = 'block';
            videoGroup.style.display = 'none';
        } else if (type === 'video_file') {
            urlGroup.style.display = 'none';
            videoGroup.style.display = 'block';
        } else {
            urlGroup.style.display = 'none';
            videoGroup.style.display = 'none';
        }
    }
    
    // Set initial state
    updateCameraInputFields();
    
    // Add change event listener
    cameraTypeSelect.addEventListener('change', updateCameraInputFields);

    
    
});

// Modify the clear-boxes button handler to also clear groups
document.getElementById('clear-boxes').addEventListener('click', () => {
    console.log("Clear boxes clicked");
    canvas.clear();
    fabricObjects = [];
    document.getElementById('box-list').innerHTML = '';
    
    // Also clear groups if grouping is initialized
    if (window.boxGroups) {
        window.boxGroups = [];
        const groupsList = document.getElementById('groups-list');
        if (groupsList) {
            groupsList.innerHTML = '<div class="no-groups">No groups created yet</div>';
        }
    }
});
// Add this function at the end of the existing admin.js file, before the window exports

// Auto-generate boxes using AI model
// Auto-generate boxes using AI model
async function autoGenerateBoxes() {
    console.log("Auto-generate boxes clicked!"); // Debug log
    
    const cameraId = document.getElementById('camera-id').value;
    if (!cameraId) {
        alert('Please register a camera first');
        return;
    }

    const button = document.getElementById('auto-generate-boxes');
    const confidenceThreshold = document.getElementById('confidence-threshold')?.value || 0.3;
    
    console.log("Starting auto-generation with confidence:", confidenceThreshold); // Debug log
    
    try {
        // Show loading state
        button.disabled = true;
        button.textContent = 'Generating...';
        
        // Get current frame from video feed
        const videoElement = document.getElementById('video-feed');
        if (!videoElement.src) {
            throw new Error('No video feed available. Please capture a frame first.');
        }
        
        console.log("Video element found, capturing frame..."); // Debug log
        
        // Create a canvas to capture the current frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = videoElement.naturalWidth || videoElement.width || 640;
        tempCanvas.height = videoElement.naturalHeight || videoElement.height || 480;
        
        // Draw the current video frame
        tempCtx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
        
        // Convert to blob
        const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));
        
        console.log("Frame captured, sending to server..."); // Debug log
        
        // Create form data
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        formData.append('confidence', confidenceThreshold);
        
        // Send to backend for AI processing
        const response = await fetch('/api/auto-generate-boxes', {
            method: 'POST',
            body: formData
        });
        
        console.log("Server response status:", response.status); // Debug log
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate boxes');
        }
        
        const result = await response.json();
        console.log("Server response:", result); // Debug log
        
        // Ask user if they want to clear existing boxes
        if (fabricObjects.length > 0) {
            const shouldClear = confirm(
                `Found ${result.boxes.length} parking spots!\n\n` +
                `You currently have ${fabricObjects.length} existing boxes.\n` +
                `Do you want to clear existing boxes and replace them?\n\n` +
                `Click "OK" to replace, or "Cancel" to add to existing boxes.`
            );
            
            if (shouldClear) {
                // Clear existing boxes
                canvas.clear();
                fabricObjects = [];
                document.getElementById('box-list').innerHTML = '';
                
                // Also clear groups if grouping is initialized
                if (window.boxGroups) {
                    window.boxGroups = [];
                    const groupsList = document.getElementById('groups-list');
                    if (groupsList) {
                        groupsList.innerHTML = '<div class="no-groups">No groups created yet</div>';
                    }
                }
            }
        }
        
        // Add generated boxes to canvas
        if (result.boxes && result.boxes.length > 0) {
            addGeneratedBoxesToCanvas(result.boxes);
            alert(`Successfully generated ${result.boxes.length} parking spot boxes!`);
        } else {
            alert('No parking spots detected. Try adjusting the confidence threshold or ensure the image contains parking areas.');
        }
        
    } catch (error) {
        console.error('Error generating boxes:', error);
        alert(`Error generating boxes: ${error.message}`);
    } finally {
        // Reset button state
        button.disabled = false;
        button.textContent = '🤖 Auto-Generate Boxes';
    }
}

// Add generated boxes to canvas (helper function)
function addGeneratedBoxesToCanvas(boxes) {
    console.log("Adding", boxes.length, "boxes to canvas"); // Debug log
    
    if (!canvas) {
        console.error('Canvas not initialized');
        return;
    }
    
    const canvasWidth = canvas.getWidth();
    const canvasHeight = canvas.getHeight();
    
    boxes.forEach((box, index) => {
        // Convert normalized coordinates to canvas coordinates
        const left = box.x1 * canvasWidth;
        const top = box.y1 * canvasHeight;
        const width = (box.x2 - box.x1) * canvasWidth;
        const height = (box.y2 - box.y1) * canvasHeight;
        
        // Get color based on group
        const groupColors = {
            'group_1': '#FF6B6B',  // Red
            'group_2': '#4ECDC4',  // Teal
            'group_3': '#45B7D1',  // Blue
            'group_4': '#96CEB4',  // Green
            'group_5': '#FFEAA7',  // Yellow
            'default': '#FF9F43'   // Orange
        };
        const color = groupColors[box.group_id] || groupColors['default'];
        
        // Create fabric rectangle
        const rect = new fabric.Rect({
            left: left,
            top: top,
            width: width,
            height: height,
            fill: 'transparent',
            stroke: color,
            strokeWidth: 2,
            transparentCorners: false,
            cornerColor: 'white',
            cornerStrokeColor: 'black',
            borderColor: 'black',
            cornerSize: 10,
            padding: 5,
            cornerStyle: 'circle',
            hasRotatingPoint: true,
            centeredRotation: true,
            angle: box.angle || 0
        });
        
        // Add metadata
        rect.groupId = box.group_id || 'default';
        rect.confidence = box.confidence || 0;
        rect.isAIGenerated = true;
        
        // Add to canvas and tracking
        canvas.add(rect);
        fabricObjects.push(rect);
    });
    
    // Update the box list
    updateBoxList();
    canvas.renderAll();
}


// Make sure functions are defined in global scope
window.autoGenerateBoxes = autoGenerateBoxes;
window.addGeneratedBoxesToCanvas = addGeneratedBoxesToCanvas;
window.updateBoxAngle = updateBoxAngle;
window.removeBox = removeBox;


