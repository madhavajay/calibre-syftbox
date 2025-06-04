const DEFAULT_SERVER_URL = "https://syftbox.net/";
const DEFAULT_SYFT_URL = "syft://";

document.addEventListener("DOMContentLoaded", () => {
    const serverUrlInput = document.getElementById("serverUrl");
    const syftUrlInput = document.getElementById("syftUrl");

    chrome.storage.sync.get(["serverUrl", "syftUrl"], (result) => {
        serverUrlInput.value = result.serverUrl || DEFAULT_SERVER_URL;
        syftUrlInput.value = result.syftUrl || DEFAULT_SYFT_URL;
    });

    document.getElementById("save").addEventListener("click", () => {
        chrome.storage.sync.set({
            serverUrl: serverUrlInput.value,
            syftUrl: syftUrlInput.value
        }, () => {
            alert("Settings saved!");
        });
    });
});
