// Grouping functionality for admin interface
// Use window.boxGroups to avoid redeclaration
window.boxGroups = window.boxGroups || []; 
window.isGroupSelectionMode = false;
window.currentGroupSelection = [];

// Initialize grouping functionality
function initGrouping(canvasRef, fabricObjectsRef) {
    console.log("Initializing grouping functionality");
    
    // Create grouping UI elements
    createGroupingUI();
    
    // Set up event listeners
    setupGroupEventListeners(canvasRef);
}

// Create the grouping UI elements
function createGroupingUI() {
    if (document.getElementById('toggle-group-mode')) {
        console.log("Grouping UI already exists");
        return;
    }
    const sidebar = document.querySelector('.sidebar');
    
    // Create grouping section
    const groupingSection = document.createElement('div');
    groupingSection.innerHTML = `
        <h3 class="section-title">Box Grouping</h3>
        <div class="controls">
            <button id="toggle-group-mode">Create Group</button>
            <button id="save-group" style="display: none;">Save Group</button>
        </div>
        <div id="selection-info" style="margin: 10px 0; padding: 5px; display: none;"></div>
        <div id="groups-list" class="box-list">
            <!-- Groups will be listed here -->
        </div>
    `;
    
    // Insert before the box list section
    const boxListSection = document.querySelector('.sidebar h3.section-title:last-of-type');
    sidebar.insertBefore(groupingSection, boxListSection);
    
    // Add CSS for group elements
    const style = document.createElement('style');
    style.textContent = `
        .group-item {
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 5px;
            border-radius: 4px;
            background-color: #f9f9f9;
        }

        .group-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }

        .group-controls {
            display: flex;
            gap: 5px;
        }

        .group-controls button {
            width: auto;
            padding: 5px 8px;
            font-size: 12px;
        }

        .group-info {
            font-size: 12px;
            color: #666;
        }
        
        #selection-info {
            background-color: #e3f2fd;
            border-radius: 4px;
            padding: 8px;
            text-align: center;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);
}

// Set up event listeners for grouping functionality
function setupGroupEventListeners(canvas) {
    const toggleBtn = document.getElementById('toggle-group-mode');
    const saveBtn = document.getElementById('save-group');
    
    if (!toggleBtn || !saveBtn) {
        console.error("Grouping buttons not found in the DOM");
        return;
    }
    
    // Remove any existing event listeners to prevent duplicates
    toggleBtn.removeEventListener('click', toggleGroupHandler);
    saveBtn.removeEventListener('click', saveGroupHandler);
    
    // Add new event listeners with proper context
    toggleBtn.addEventListener('click', toggleGroupHandler);
    saveBtn.addEventListener('click', saveGroupHandler);
    
    function toggleGroupHandler() {
        console.log("Toggle group mode clicked");
        toggleGroupSelectionMode(canvas);
    }
    
    function saveGroupHandler() {
        console.log("Save group clicked");
        saveGroup(canvas);
    }
}

// Toggle group selection mode
function toggleGroupSelectionMode(canvas) {
    if (window.isDrawMode) {
        alert('Please stop drawing before creating groups');
        return;
    }
    window.isGroupSelectionMode = !window.isGroupSelectionMode;
    console.log("Group selection mode:", window.isGroupSelectionMode);
    
    // Update button text
    const groupButton = document.getElementById('toggle-group-mode');
    const saveGroupBtn = document.getElementById('save-group');
    const selectionInfo = document.getElementById('selection-info');
    
    if (window.isGroupSelectionMode) {
        // Clear any previous group selection
        window.currentGroupSelection = [];
        
        // Change button styling and text
        groupButton.textContent = 'Cancel Grouping';
        groupButton.style.backgroundColor = '#d32f2f';
        
        // Show the save group button and selection info
        saveGroupBtn.style.display = 'block';
        selectionInfo.style.display = 'block';
        selectionInfo.textContent = 'Click on boxes to select them for the group';
        
        // Set up canvas for selection
        canvas.selection = true;
        canvas.discardActiveObject();
        canvas.forEachObject(function(obj) {
            obj.selectable = true;
            obj.evented = true;
        });
        
        alert('Select boxes to group together, then click "Save Group"');
    } else {
        // Reset button state
        groupButton.textContent = 'Create Group';
        groupButton.style.backgroundColor = '#4CAF50';
        
        // Hide the save group button and selection info
        saveGroupBtn.style.display = 'none';
        selectionInfo.style.display = 'none';
        
        // Clear selection highlights
        clearGroupSelection(canvas);
    }
    
    canvas.renderAll();
}

// Clear group selection
function clearGroupSelection(canvas) {
    // Reset all objects to their original appearance
    window.currentGroupSelection.forEach(obj => {
        obj.set({
            stroke: obj.originalStroke || 'green',
            strokeWidth: 2
        });
    });
    
    window.currentGroupSelection = [];
    canvas.renderAll();
}

// Save a group
function saveGroup(canvas) {
    console.log("Saving group with selection:", window.currentGroupSelection);
    
    if (window.currentGroupSelection.length === 0) {
        alert('Please select at least one box to create a group');
        return;
    }
    
    // Prompt for group location
    const groupLocation = prompt('Enter a location name for this group:');
    if (!groupLocation) {
        return; // User cancelled
    }
    
    // Create a new group
    const newGroup = {
        id: Date.now(), // Use timestamp for unique ID
        location: groupLocation,
        boxes: [...window.currentGroupSelection]
    };
    
    window.boxGroups.push(newGroup);
    console.log("Group created:", newGroup);
    
    // Update box properties to associate with group
    window.currentGroupSelection.forEach(obj => {
        obj.groupId = newGroup.id;
        obj.groupLocation = groupLocation;
        
        // Change color to indicate group membership
        if (!obj.hasOwnProperty('originalStroke')) {
            obj.originalStroke = obj.stroke;
        }
        
        // Use a different color for each group (cycle through a few colors)
        const groupColors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33'];
        const colorIndex = window.boxGroups.length % groupColors.length;
        obj.set({
            stroke: groupColors[colorIndex],
            strokeWidth: 2
        });
    });
    
    // Exit group selection mode
    toggleGroupSelectionMode(canvas);
    
    // Update the groups list
    updateGroupsList();
    
    canvas.renderAll();
}

// Update the groups list in the UI
function updateGroupsList() {
    const groupsList = document.getElementById('groups-list');
    if (!groupsList) return;
    
    groupsList.innerHTML = '';
    
    if (window.boxGroups.length === 0) {
        groupsList.innerHTML = '<div class="no-groups">No groups created yet</div>';
        return;
    }
    
    window.boxGroups.forEach(group => {
        const groupItem = document.createElement('div');
        groupItem.className = 'group-item';
        
        groupItem.innerHTML = `
            <div class="group-header">
                <span>Group: ${group.location}</span>
                <div class="group-controls">
                    <button onclick="editGroup(${group.id})">Edit</button>
                    <button onclick="deleteGroup(${group.id})">Delete</button>
                </div>
            </div>
            <div class="group-info">
                ${group.boxes.length} boxes
            </div>
        `;
        
        groupsList.appendChild(groupItem);
    });
}

// Handle object selection for grouping
function handleObjectSelection(obj, canvas) {
    if (!window.isGroupSelectionMode) return false;
    
    console.log("Object selected in group mode:", obj);
    
    // Check if object is already in selection
    const index = window.currentGroupSelection.findIndex(o => o === obj);
    
    if (index === -1) {
        // Add to selection
        window.currentGroupSelection.push(obj);
        // Store original color
        if (!obj.hasOwnProperty('originalStroke')) {
            obj.originalStroke = obj.stroke;
        }
        // Change stroke color to indicate selection
        obj.set({
            stroke: 'purple',
            strokeWidth: 3
        });
    } else {
        // Remove from selection
        window.currentGroupSelection.splice(index, 1);
        // Restore original color
        obj.set({
            stroke: obj.originalStroke || 'green',
            strokeWidth: 2
        });
    }
    
    // Update selection info
    const selectionInfo = document.getElementById('selection-info');
    if (selectionInfo) {
        selectionInfo.textContent = `${window.currentGroupSelection.length} boxes selected`;
    }
    
    canvas.renderAll();
    return true; // Indicate that we handled the selection
}

// Edit a group
function editGroup(groupId) {
    const canvas = window.canvas; // Get canvas from global scope
    const group = window.boxGroups.find(g => g.id === groupId);
    if (!group) return;
    
    const newLocation = prompt('Enter a new location name:', group.location);
    if (!newLocation) return;
    
    group.location = newLocation;
    
    // Update all boxes in the group
    group.boxes.forEach(obj => {
        obj.groupLocation = newLocation;
    });
    
    updateGroupsList();
    canvas.renderAll();
}

// Delete a group
function deleteGroup(groupId) {
    const canvas = window.canvas; // Get canvas from global scope
    const group = window.boxGroups.find(g => g.id === groupId);
    if (!group) return;
    
    const confirmed = confirm(`Are you sure you want to delete group "${group.location}"?`);
    if (!confirmed) return;
    
    // Reset all boxes in the group
    group.boxes.forEach(obj => {
        obj.set({
            stroke: obj.originalStroke || 'green',
            strokeWidth: 2
        });
        delete obj.groupId;
        delete obj.groupLocation;
        delete obj.originalStroke;
    });
    
    // Remove group from the list
    window.boxGroups = window.boxGroups.filter(g => g.id !== groupId);
    
    updateGroupsList();
    canvas.renderAll();
}

// Export functions to global scope
window.initGrouping = initGrouping;
window.toggleGroupSelectionMode = toggleGroupSelectionMode;
window.saveGroup = saveGroup;
window.editGroup = editGroup;
window.deleteGroup = deleteGroup;
window.handleObjectSelection = handleObjectSelection;
