// do not move these variables out of this function
function getGrokSelectors() {
    const grokSubmitButton = document.querySelector('button[type="submit"]');
    const grokTextarea = document.querySelector('textarea[aria-label="Ask Grok anything"]');
    const grokForm = document.querySelector('form.bottom-0.w-full.text-base.flex.flex-col.gap-2.items-center.justify-center.relative.z-10');
    const grokButtonContainer = grokSubmitButton?.closest('div.flex');
    return {
        submitButton: grokSubmitButton,
        textarea: grokTextarea,
        form: grokForm,
        buttonContainer: grokButtonContainer
    };
}

// Grok-specific button placement
function placeButtonForGrok(buttonContainer, submitButton, customButton) {
    // For Grok, insert before the submit button in the button group
    const buttonGroup = submitButton?.parentNode;
    if (buttonGroup && buttonGroup.contains(submitButton)) {
        buttonGroup.insertBefore(customButton, submitButton);
    } else if (buttonContainer && buttonContainer.contains(submitButton)) {
        buttonContainer.insertBefore(customButton, submitButton);
    } else {
        console.log('Submit button is not a child of the expected container.');
    }
}

// Grok-specific spinner handling functions
function attachSpinnerForGrok(textarea) {
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

    // Find the textarea's parent container that has position relative
    const textareaContainer = textarea.closest('.flex-col') || textarea.parentElement;

    if (textareaContainer) {
        // Make sure the container has position relative for absolute positioning to work
        const originalPosition = window.getComputedStyle(textareaContainer).position;
        if (originalPosition !== 'relative' && originalPosition !== 'absolute') {
            textareaContainer.classList.add('mcp-relative-container');
        }

        textareaContainer.appendChild(overlay);
        return overlay;
    }

    return null;
}

function removeSpinnerForGrok() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Grok-specific textarea handling functions
function populateGrokTextarea(newText) {
    const textarea = document.querySelector('textarea[aria-label="Ask Grok anything"]');
    if (!textarea) return;

    textarea.value = newText;
    textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    textarea.focus();
}

// Export Grok-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.grok = {
    selectors: getGrokSelectors,
    placeButton: placeButtonForGrok,
    attachSpinner: attachSpinnerForGrok,
    removeSpinner: removeSpinnerForGrok,
    populateTextarea: populateGrokTextarea
}; 