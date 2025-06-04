// do not move these variables out of this function
function getDeepSeekSelectors() {
    const deepSeekSubmitButton = document.querySelector('div[role="button"]._7436101');
    const deepSeekTextarea = document.querySelector('textarea#chat-input');
    const deepSeekForm = document.querySelector('div._77cefa5');
    const deepSeekButtonContainer = deepSeekSubmitButton?.closest('div.ec4f5d61');
    return {
        submitButton: deepSeekSubmitButton,
        textarea: deepSeekTextarea,
        form: deepSeekForm,
        buttonContainer: deepSeekButtonContainer
    };
}
// Function to populate the textarea with given text
function populateDeepSeekTextarea(text) {
    console.log('Populating DeepSeek textarea with text:', text);
    const selectors = getDeepSeekSelectors();
    const textarea = selectors.textarea;
    if (textarea) {
        textarea.value = text;
        textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    } else {
        console.error('Textarea not found for DeepSeek.');
    }
}

// DeepSeek-specific button placement
function placeButtonForDeepSeek(buttonContainer, submitButton, customButton) {
    // For DeepSeek, insert before the submit button in the button group
    const buttonGroup = submitButton?.parentNode;
    if (buttonGroup && buttonGroup.contains(submitButton)) {
        buttonGroup.insertBefore(customButton, submitButton);
    } else if (buttonContainer && buttonContainer.contains(submitButton)) {
        buttonContainer.insertBefore(customButton, submitButton);
    } else {
        console.log('Submit button is not a child of the expected container.');
    }
}

// DeepSeek-specific spinner handling functions
function attachSpinnerForDeepSeek(textarea) {
    // Check if a spinner already exists
    const existingSpinner = document.querySelector('.mcp-textarea-overlay');
    if (existingSpinner) return existingSpinner;

    // Create a targeted overlay for just the textarea
    const overlay = document.createElement('div');
    overlay.className = 'mcp-textarea-overlay';

    // Create the spinner
    const spinner = document.createElement('div');
    spinner.className = 'mcp-textarea-spinner';

    overlay.appendChild(spinner);

    // Find the top-level container of the DeepSeek UI
    const topLevelContainer = textarea.closest('div._77cefa5');

    if (topLevelContainer) {
        // Make sure the container has position relative for absolute positioning to work
        const originalPosition = window.getComputedStyle(topLevelContainer).position;
        if (originalPosition !== 'relative' && originalPosition !== 'absolute') {
            topLevelContainer.classList.add('mcp-relative-container');
        }

        topLevelContainer.appendChild(overlay);
        return overlay;
    }

    return null;
}

function removeSpinnerForDeepSeek() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Export DeepSeek-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.deepseek = {
    selectors: getDeepSeekSelectors,
    placeButton: placeButtonForDeepSeek,
    attachSpinner: attachSpinnerForDeepSeek,
    removeSpinner: removeSpinnerForDeepSeek,
    populateTextarea: populateDeepSeekTextarea
}; 