// do not move these variables out of this function
function getClaudeSelectors() {
    const claudeSubmitButton = document.querySelector('button[aria-label="Send message"]');
    const claudeTextarea = document.querySelector('div[contenteditable="true"].ProseMirror');
    const claudeForm = document.querySelector('div.flex.flex-col.bg-bg-000.border-0\\.5.border-border-300.mx-2.md\\:mx-0.items-stretch');
    const claudeButtonContainer = claudeSubmitButton?.closest('div.flex.gap-2\\.5.w-full.items-center');
    return {
        submitButton: claudeSubmitButton,
        textarea: claudeTextarea,
        form: claudeForm,
        buttonContainer: claudeButtonContainer
    };
}

// Function to populate the textarea with given text
function populateClaudeTextarea(text) {
    console.log('Populating Claude textarea with text:', text);
    const selectors = getClaudeSelectors();
    const textarea = selectors.textarea;
    if (textarea) {
        textarea.innerHTML = text;
        textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    } else {
        console.error('Textarea not found for Claude.');
    }
}

// Claude-specific button placement
function placeButtonForClaude(buttonContainer, submitButton, customButton) {
    // For Claude, insert before the submit button in the button group
    const buttonGroup = submitButton?.parentNode;
    if (buttonGroup && buttonGroup.contains(submitButton)) {
        buttonGroup.insertBefore(customButton, submitButton);
    } else if (buttonContainer && buttonContainer.contains(submitButton)) {
        buttonContainer.insertBefore(customButton, submitButton);
    } else {
        console.log('Submit button is not a child of the expected container.');
    }
}

// Claude-specific spinner handling functions
function attachSpinnerForClaude(textarea) {
    // Check if a spinner already exists
    const existingSpinner = document.querySelector('.mcp-textarea-overlay');
    if (existingSpinner) return;

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

function removeSpinnerForClaude() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Export Claude-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.claude = {
    selectors: getClaudeSelectors,
    placeButton: placeButtonForClaude,
    populateTextarea: populateClaudeTextarea,
    attachSpinner: attachSpinnerForClaude,
    removeSpinner: removeSpinnerForClaude
}; 