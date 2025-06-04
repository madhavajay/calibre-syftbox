var SERVER_URL = "https://syftboxdev.openmined.org/";
var SYFT_URL = "syft://";

function getStorageData(keys) {
    return new Promise((resolve, reject) => {
        chrome.storage.sync.get(keys, (result) => {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else {
                resolve(result);
            }
        });
    });
}

async function initializeConfig() {
    try {
        const serverResult = await getStorageData(["serverUrl"]);
        console.log("Using server:", serverResult.serverUrl);
        SERVER_URL = serverResult.serverUrl;

        const syftResult = await getStorageData(["syftUrl"]);
        console.log("Using syftUrl:", syftResult.syftUrl);
        SYFT_URL = syftResult.syftUrl;
    } catch (error) {
        console.error("Error retrieving storage data:", error);
    }
}

initializeConfig().then(() => {
    // Continue with the rest of the code that depends on the configuration

    // Configuration
    const CONFIG = {
        serverUrl: SERVER_URL,
        syftUrl: SYFT_URL,
        buttonText: "MCP +",
        autoRun: true
    };

    const syftboxSdkInstance = new SyftBoxSDK({ serverUrl: CONFIG.serverUrl });
    syftFetch.serverUrl = CONFIG.serverUrl;
    console.log(">>>>> syftFetch", syftFetch.serverUrl)

    try {
        console.log(">>>>> CONFIG.syftUrl", CONFIG.syftUrl)
        const parsedUrl = syftboxSdkInstance.parseSyftUrl(CONFIG.syftUrl);
    } catch (error) {
        if (error.message === "Invalid syft URL format") {
            alert("Please configure your syft:// URL in the extension settings.");
        } else {
            console.error("An unexpected error occurred:", error);
        }
    }

    // Site detection functions
    function isChatGPT() {
        return window.location.hostname === 'chat.openai.com' || window.location.hostname === 'chatgpt.com';
    }

    function isClaude() {
        return window.location.hostname === 'claude.ai';
    }

    function isGrok() {
        return window.location.hostname === 'grok.com';
    }

    function isPerplexity() {
        return window.location.hostname === 'perplexity.ai' || window.location.hostname === 'www.perplexity.ai';
    }

    function isCopilot() {
        return window.location.hostname === 'copilot.microsoft.com';
    }

    function isDeepSeek() {
        return window.location.hostname === 'chat.deepseek.com';
    }

    function isGemini() {
        return window.location.hostname === 'gemini.google.com';
    }

    // Don't run multiple times
    if (window._mcpButtonAdded) {
        console.log("MCP button already added, skipping initialization");
    } else {
        window._mcpButtonAdded = true;
    }
    window._mcpButtonAdded = true;

    // Clean up any existing buttons first
    const existingButtons = document.querySelectorAll('.mcp-button');
    existingButtons.forEach(button => button.remove());

    // Create and add styles
    const style = document.createElement('link');
    style.rel = 'stylesheet';
    const hostname = window.location.hostname;
    let styleFile = 'styles-chatgpt.css';

    if (isClaude()) {
        styleFile = 'styles-claude.css';
    } else if (isGrok()) {
        styleFile = 'styles-grok.css';
    } else if (isPerplexity()) {
        styleFile = 'styles-perplexity.css';
    } else if (isCopilot()) {
        styleFile = 'styles-copilot.css';
    } else if (isDeepSeek()) {
        styleFile = 'styles-deepseek.css';
    } else if (isGemini()) {
        styleFile = 'styles-gemini.css';
    }

    style.href = chrome.runtime.getURL(styleFile);
    document.head.appendChild(style);

    // Utility functions
    function encodeInputText(text) {
        return encodeURIComponent(text.trim());
    }

    // Helper function to get selectors for current site
    function getConfigForCurrentSite() {
        const hostname = window.location.hostname;
        console.log(`Current hostname: ${hostname}`);
        let site;

        if (isPerplexity()) {
            site = 'perplexity';
        } else if (isDeepSeek()) {
            site = 'deepseek';
        } else {
            site = hostname.split('.')[0];
        }

        console.log(`Selected site: ${site}`);
        const siteConfig = window.MCP_SITES?.[site] || null;
        console.log(`MCP_SITES configuration for ${site}:`, siteConfig);
        return siteConfig;
    }

    function getSelectorsForCurrentSite() {
        const config = getConfigForCurrentSite();
        return config ? config.selectors() : null;
    }

    // Updated findTargetElements to use the selector variables
    function findTargetElements() {
        const selectors = getSelectorsForCurrentSite();

        if (!selectors) {
            console.log('No selectors found for site:', window.location.hostname);
            return null;
        }
        let submitButton, buttonContainer;

        // Use the actual selector objects from the site config
        submitButton = selectors.submitButton;
        buttonContainer = selectors.buttonContainer;

        if (!buttonContainer || !submitButton) {
            console.log('Button container or submit button not found for', window.location.hostname);
            return null;
        }

        return { submitButton, buttonContainer };
    }


    // function setPromptText(newText) {
    //     const textarea = getPromptTextareaElement();
    //     if (!textarea) return;

    //     if (isClaude()) {
    //         textarea.innerHTML = `<p>${newText.replace(/\n/g, "</p><p>")}</p>`;
    //         textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    //     } else if (isGemini()) {
    //         textarea.textContent = newText;
    //     } else {
    //         textarea.value = newText;
    //         textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    //     }
    //     textarea.focus();
    // }

    // Updated findFormElement to use the selector variables
    function findFormElement(hostname) {
        const selectors = getSelectorsForCurrentSite();
        return selectors ? document.querySelector(selectors.form) : null;
    }

    // Overlay and spinner handling
    function createOverlay(hostname) {
        const overlay = document.createElement('div');
        overlay.className = `mcp-overlay mcp-overlay-${hostname.split('.')[0]}`;

        const spinner = document.createElement('div');
        spinner.className = `mcp-spinner mcp-spinner-${hostname.split('.')[0]}`;

        overlay.appendChild(spinner);
        return overlay;
    }

    function removeOverlay(form) {
        const overlay = form?.querySelector('.mcp-overlay');
        if (overlay) overlay.remove();
    }

    // Button creation and handling
    function createCustomButton() {
        const button = document.createElement('button');
        button.className = 'mcp-button';
        button.innerHTML = `
        <span class="mcp-button-text">${CONFIG.buttonText}</span>
        <div class="mcp-spinner" style="display: none;"></div>
    `;

        let alreadyClicked = false;

        button.onclick = async function (event) {
            event.preventDefault();
            if (alreadyClicked) return;
            alreadyClicked = true;
            console.log("MCP button clicked");

            // Show loading state
            const spinner = button.querySelector('.mcp-spinner');
            spinner.style.display = 'block';
            button.querySelector('.mcp-button-text').style.display = 'none';

            const selectors = getSelectorsForCurrentSite();
            const textarea = selectors?.textarea;
            if (!textarea) {
                console.error("Textarea not found!");
                alreadyClicked = false;
                spinner.style.display = 'none';
                button.querySelector('.mcp-button-text').style.display = 'block';
                return;
            }

            let text = textarea.value?.trim() || textarea.textContent.trim();

            if (!text) {
                console.error("No text found in textarea!");
                alreadyClicked = false;
                spinner.style.display = 'none';
                button.querySelector('.mcp-button-text').style.display = 'block';
                return;
            }

            console.log("syftUrl", CONFIG.syftUrl);

            const body = { "model": "openai/chatgpt-4o-latest", "messages": [{ "role": "user", "content": text }], "temperature": 1.0, "max_tokens": 256 }
            const bodyText = JSON.stringify(body);
            console.log(">>>> body", body, bodyText)

            // const url = `${CONFIG.serverUrl}?q=${encodeInputText(text)}`;
            // const instruction = `Instruction: Here is some context I want to chat about in the next message, respond only with the words "OK".`;
            const instruction = `Answer this question: ${text}. Using only this context:`;
            // const instruction = text
            // Always use the config's attachSpinner method
            const config = getConfigForCurrentSite();
            if (config && config.attachSpinner) {
                config.attachSpinner(textarea);
            }

            const headers = {}

            try {
                console.log("Fetching data from:", CONFIG.syftUrl);
                const response = await syftFetch(CONFIG.syftUrl, {
                    method: 'POST',
                    headers,
                    body: bodyText
                });
                console.log(">>> response", response)
                const responseData = response
                // const response = await fetch(url);
                // if (!response.ok) {
                //     throw new Error(`HTTP error! status: ${response.status}`);
                // }

                // const responseData = await response.json();
                console.log("responseData", responseData)
                const data = responseData.data.message.body.choices[0].message.content;
                console.log("Received data:", data);

                const message = `${instruction}\n\n${data}`;
                config.populateTextarea(message);

                // Wait for the text to be set before clicking submit
                setTimeout(() => {
                    const elements = findTargetElements();
                    if (elements?.submitButton) {
                        console.log("Clicking submit button");
                        if (isDeepSeek()) {
                            elements.submitButton.dispatchEvent(new MouseEvent('click', {
                                bubbles: true,
                                cancelable: true,
                                view: window
                            }));
                        } else {
                            elements.submitButton.click();
                        }
                    } else {
                        console.error("Submit button not found");
                    }
                }, 1000);
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to process the request. Please try again.');
            } finally {
                // Always use the config's removeSpinner method
                if (config && config.removeSpinner) {
                    config.removeSpinner();
                }

                setTimeout(() => {
                    alreadyClicked = false;
                    spinner.style.display = 'none';
                    button.querySelector('.mcp-button-text').style.display = 'block';
                }, 2000);
            }
        };

        return button;
    }


    // Main button management
    function addMCPButton() {
        const elements = findTargetElements();
        if (!elements) {
            console.log('Target elements not found, retrying...');
            setTimeout(addMCPButton, 1000);
            return;
        }

        const { buttonContainer, submitButton } = elements;

        // Remove any existing MCP buttons
        const existingButtons = document.querySelectorAll('.mcp-button');
        existingButtons.forEach(button => button.remove());

        const customButton = createCustomButton();
        const siteConfig = getConfigForCurrentSite();

        if (siteConfig?.placeButton) {
            siteConfig.placeButton(buttonContainer, submitButton, customButton);
        } else {
            // Default placement if no site-specific function exists
            console.log('Default button placement');
            buttonContainer.appendChild(customButton);
        }
    }

    // Initialization and setup
    function initialize() {
        const site = window.location.hostname.split('.')[0];
        const siteConfig = window.MCP_SITES?.[site];

        if (!siteConfig) {
            console.log('No configuration found for site:', site);
            return;
        }

        const elements = findTargetElements();
        if (!elements) {
            console.log('Target elements not found for', site);
            return;
        }

        const { submitButton, buttonContainer } = elements;
        const customButton = createCustomButton();

        // Use site-specific button placement
        if (siteConfig.placeButton) {
            siteConfig.placeButton(buttonContainer, submitButton, customButton);
        } else {
            // Default button placement
            buttonContainer.insertBefore(customButton, submitButton);
        }

        // Set up button observer for Gemini
        if (isGemini() && siteConfig.setupButtonObserver) {
            siteConfig.setupButtonObserver();
        }

        console.log("MCP button added to", site);
    }

    function setupObserver() {
        console.log("Setting up MutationObserver for dynamic loading...");
        const observer = new MutationObserver(function (mutations) {
            if (!document.querySelector('.mcp-button')) {
                console.log("MCP button not found, adding it...");
                addMCPButton();
            }
            const elements = findTargetElements();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true
        });
    }

    // Add spinner animation style
    const spinnerStyle = document.createElement('style');
    spinnerStyle.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
    document.head.appendChild(spinnerStyle);

    // Export functions for testing
    window.MCP = {
        isChatGPT,
        isClaude,
        isGrok,
        isPerplexity,
        isCopilot,
        isDeepSeek,
        isGemini,
        findTargetElements,
        createCustomButton,
        addMCPButton,
        initialize,
        setupObserver,
    };

    // Auto-run if enabled
    function init() {
        if (CONFIG.autoRun) {
            initialize();
            setupObserver();
            console.log("MCP button setup complete.");
        } else {
            console.log("MCP button setup is disabled. Use the context menu to trigger it.");
        }
    }

    // Initialize only if autoRun is true
    if (CONFIG.autoRun) {
        init();
    } else {
        console.log("MCP button setup is disabled. Use the context menu to trigger it.");
    }

}).catch(error => {
    console.error("Failed to initialize configuration:", error);
});
