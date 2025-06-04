// do not move these variables out of this function
function getGeminiSelectors() {
    let geminiSubmitButton = Array.from(document.querySelectorAll('button.send-button')).find(button => button.offsetParent !== null);
    if (geminiSubmitButton) {
        console.log('Found geminiSubmitButton:', geminiSubmitButton);
        console.log('Parent of geminiSubmitButton:', geminiSubmitButton.parentElement);
    } else {
        geminiSubmitButton = Array.from(document.querySelectorAll('button.speech_dictation_mic_button')).find(button => button.offsetParent !== null);
        if (geminiSubmitButton) {
            console.log('Found geminiSubmitButton (speech dictation):', geminiSubmitButton);
            console.log('Parent of geminiSubmitButton (speech dictation):', geminiSubmitButton.parentElement);
        }
    }

    const geminiTextarea = document.querySelector('div[contenteditable="true"].textarea.new-input-ui');
    const geminiForm = document.querySelector('form');
    const geminiButtonContainer = geminiSubmitButton?.closest('div.input-buttons-wrapper-bottom');
    return {
        submitButton: geminiSubmitButton,
        textarea: geminiTextarea,
        form: geminiForm,
        buttonContainer: geminiButtonContainer
    };
}

// Function to populate the textarea with given text
function populateGeminiTextarea(text) {
    console.log('Populating Gemini textarea with text:', text);
    const selectors = getGeminiSelectors();
    const textarea = selectors.textarea;
    if (textarea) {
        textarea.textContent = text;
        textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    } else {
        console.error('Textarea not found for Gemini.');
    }
}

// Gemini-specific button placement
function placeButtonForGemini(buttonContainer, submitButton, customButton) {
    // For Gemini, insert the custom button to the left of the button group on the same level
    const inputAreaContainer = document.querySelector('div.input-area-container');
    if (inputAreaContainer) {
        inputAreaContainer.insertBefore(customButton, inputAreaContainer.firstChild);
        console.log('Custom button added to the left of the button group.');
    } else {
        console.error('Could not find the input area container for Gemini.');
    }
}

// Gemini-specific spinner handling functions
function attachSpinnerForGemini(textarea) {
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

    // Find the top-level container of the Gemini UI
    const topLevelContainer = textarea.closest('div.input-buttons-wrapper-bottom');

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

function removeSpinnerForGemini() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Export Gemini-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.gemini = {
    selectors: getGeminiSelectors,
    placeButton: placeButtonForGemini,
    attachSpinner: attachSpinnerForGemini,
    removeSpinner: removeSpinnerForGemini,
    populateTextarea: populateGeminiTextarea
}; 