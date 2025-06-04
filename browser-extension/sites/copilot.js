// do not move these variables out of this function
function getCopilotSelectors() {
    const copilotSubmitButton = document.querySelector('button[data-testid="submit-button"], button[aria-label="Talk to Copilot"]');
    const copilotTextarea = document.querySelector('textarea#userInput[placeholder="Message Copilot"]');
    const copilotForm = document.querySelector('div.pointer-events-auto.relative.flex.flex-col.contrast-more\\:border-2[data-testid="composer-content"]');
    const copilotButtonContainer = copilotSubmitButton?.closest('div.flex');
    return {
        submitButton: copilotSubmitButton,
        textarea: copilotTextarea,
        form: copilotForm,
        buttonContainer: copilotButtonContainer
    };
}

// Function to populate the textarea with given text
function populateCopilotTextarea(text) {
    console.log('Populating Copilot textarea with text:', text);
    const selectors = getCopilotSelectors();
    const textarea = selectors.textarea;
    if (textarea) {
        textarea.value = text;
        textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    } else {
        console.error('Textarea not found for Copilot.');
    }
}
// Copilot-specific button placement
function placeButtonForCopilot(buttonContainer, submitButton, customButton) {
    // For Copilot, try to find the button group first
    const buttonGroup = submitButton?.closest('div.flex.gap-2');
    if (buttonGroup) {
        buttonGroup.insertBefore(customButton, buttonGroup.firstChild);
        console.log('Custom button added to the left of the button group.');
    } else if (buttonContainer) {
        buttonContainer.insertBefore(customButton, buttonContainer.firstChild);
        console.log('Custom button added to the left of the button container.');
    } else {
        console.error('Could not find a suitable container for the custom button.');
    }
}

// Copilot-specific spinner handling functions
function attachSpinnerForCopilot(textarea) {
    // Check if a spinner already exists
    const existingSpinner = document.querySelector('.mcp-textarea-overlay');
    if (existingSpinner) return existingSpinner;

    // Create a targeted overlay for the entire composer
    const overlay = document.createElement('div');
    overlay.className = 'mcp-textarea-overlay';

    // Create the spinner
    const spinner = document.createElement('div');
    spinner.className = 'mcp-textarea-spinner';

    overlay.appendChild(spinner);

    // Find the top-level container of the Copilot UI
    const topLevelContainer = document.querySelector('.relative.max-h-full.min-h-composer.min-w-16.max-w-chat.rounded-5xl');

    if (topLevelContainer) {
        // Make sure the container has position relative for absolute positioning to work
        const originalPosition = window.getComputedStyle(topLevelContainer).position;
        if (originalPosition !== 'relative' && originalPosition !== 'absolute') {
            topLevelContainer.classList.add('mcp-relative-container');
        }

        topLevelContainer.appendChild(overlay);
        return overlay;
    } else {
        // Fallback to the old behavior if top container isn't found
        const textareaContainer = textarea.closest('.grid-rows-1fr-auto') || textarea.parentElement;

        if (textareaContainer) {
            const originalPosition = window.getComputedStyle(textareaContainer).position;
            if (originalPosition !== 'relative' && originalPosition !== 'absolute') {
                textareaContainer.classList.add('mcp-relative-container');
            }

            textareaContainer.appendChild(overlay);
            return overlay;
        }
    }

    return null;
}

function removeSpinnerForCopilot() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Export Copilot-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.copilot = {
    selectors: getCopilotSelectors,
    placeButton: placeButtonForCopilot,
    attachSpinner: attachSpinnerForCopilot,
    removeSpinner: removeSpinnerForCopilot,
    populateTextarea: populateCopilotTextarea
}; 