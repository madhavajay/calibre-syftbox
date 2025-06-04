chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "open-custom-options",
        title: "SyftBox Settings",
        contexts: ["action"]
    });
});

chrome.contextMenus.onClicked.addListener((info) => {
    if (info.menuItemId === "open-custom-options") {
        chrome.tabs.create({ url: chrome.runtime.getURL("options.html") });
    }
});
