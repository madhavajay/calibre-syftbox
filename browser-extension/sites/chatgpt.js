// do not move these variables out of this function
function getChatGPTSelectors() {
    const chatGPTSubmitButton = document.querySelector('button[data-testid="composer-speech-button"], button#composer-submit-button');
    const chatGPTTextarea = document.querySelector('#prompt-textarea');
    const chatGPTForm = document.querySelector('form[data-type="unified-composer"]');
    const chatGPTButtonContainer = chatGPTSubmitButton?.closest('div.w-full');
    return {
        submitButton: chatGPTSubmitButton,
        textarea: chatGPTTextarea,
        form: chatGPTForm,
        buttonContainer: chatGPTButtonContainer
    };
}

// Function to populate the textarea with given text
function populateChatGPTTextarea(text) {
    console.log('Populating ChatGPT textarea with text:', text);
    const selectors = getChatGPTSelectors();
    const textarea = selectors.textarea;
    if (textarea) {
        textarea.innerHTML = text;
        textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    } else {
        console.error('Textarea not found for ChatGPT.');
    }
}

// ChatGPT-specific button placement
function placeButtonForChatGPT(buttonContainer, submitButton, customButton) {
    // Create a container for our button with absolute positioning
    const customButtonWrapper = document.createElement('div');
    customButtonWrapper.className = 'mcp-button-container';
    customButtonWrapper.style.position = 'absolute';
    customButtonWrapper.style.right = '48px'; // Position to the left of the voice button
    customButtonWrapper.style.bottom = '9px'; // Align with other buttons
    customButtonWrapper.style.zIndex = '10'; // Ensure it's above other elements

    // Add our button to this wrapper
    customButtonWrapper.appendChild(customButton);

    // Find the form element which is the main container
    const formElement = document.querySelector('form[data-type="unified-composer"]');
    if (formElement) {
        formElement.appendChild(customButtonWrapper);
        console.log('Button placed with absolute positioning in form');
        return;
    }

    // Alternative - find the parent container that has position: relative
    const relativeContainer = document.querySelector('.relative.flex.w-full.items-end');
    if (relativeContainer) {
        relativeContainer.appendChild(customButtonWrapper);
        console.log('Button placed with absolute positioning in relative container');
        return;
    }

    // Last resort - just append to the body
    if (submitButton && submitButton.closest('form')) {
        submitButton.closest('form').appendChild(customButtonWrapper);
        console.log('Button placed with absolute positioning in form container');
    } else {
        console.error('Could not find suitable container for button placement');
    }
}

// ChatGPT-specific spinner handling functions
function attachSpinnerForChatGPT(textarea) {
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

function removeSpinnerForChatGPT() {
    const spinner = document.querySelector('.mcp-textarea-overlay');
    if (spinner) spinner.remove();
}

// Export ChatGPT-specific code
window.MCP_SITES = window.MCP_SITES || {};
window.MCP_SITES.chatgpt = {
    selectors: getChatGPTSelectors,
    placeButton: placeButtonForChatGPT,
    populateTextarea: populateChatGPTTextarea,
    attachSpinner: attachSpinnerForChatGPT,
    removeSpinner: removeSpinnerForChatGPT
}; 