// do not move these variables out of this function
function getPerplexitySelectors() {
    const perplexitySubmitButton = document.querySelector('button[aria-label="Submit"]') ||
        document.querySelector('button[aria-label="Dictation"]');
    const perplexityTextarea = document.querySelector('textarea[aria-label="Ask anything"]') ||
        document.querySelector('textarea[placeholder*="Ask anything"]');
    const perplexityForm = document.querySelector('div.rounded-3xl');
    const perplexityButtonContainer = document.querySelector('div.flex.items-center.gap-2') ||
        perplexitySubmitButton?.closest('div.flex');
    return {
        submitButton: perplexitySubmitButton,
        textarea: perplexityTextarea,
        form: perplexityForm,
        buttonContainer: perplexityButtonContainer
    };
}

// Perplexity-specific button placement
function placeButtonForPerplexity(buttonContainer, submitButton, customButton) {
    // For Perplexity, insert before the submit button
    if (buttonContainer) {
        console.log('Button container found.');
        if (submitButton) {
            console.log('Submit button found.');
            // Get the parent container of the submit button
            const submitParent = submitButton.parentElement;
            // Insert before the submit button's container
            submitParent.before(customButton);
        } else {
            console.log('Submit button not found. Adding to the end of the container.');
            // Fallback - just add to the end of the container
            buttonContainer.appendChild(customButton);
        }
    } else {
        console.log('Button container not found.');
    }
}

// Perplexity-specific spinner handling functions
function attachSpinnerForPerplexity(textarea) {
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
    const textareaContainer = textarea.closest('.grid-rows-1fr-auto') || textarea.parentElement;

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

function removeSpinnerForPerplexity() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Perplexity-specific textarea handling functions
function populatePerplexityTextarea(newText) {
    const textarea = document.querySelector('textarea[aria-label="Ask anything"]') || document.querySelector('textarea[placeholder*="Ask anything"]');
    if (!textarea) return;

    textarea.value = newText;
    textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    textarea.focus();
}

// Export Perplexity-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.perplexity = {
    selectors: getPerplexitySelectors,
    placeButton: placeButtonForPerplexity,
    attachSpinner: attachSpinnerForPerplexity,
    removeSpinner: removeSpinnerForPerplexity,
    populateTextarea: populatePerplexityTextarea
};