const STORAGE_KEYS = {
  recentChats: "recentChats"
};

const chatInput = document.getElementById("chatInput");
const saveButton = document.getElementById("saveChats");
const clearButton = document.getElementById("clearChats");
const statusMessage = document.getElementById("statusMessage");

document.addEventListener("DOMContentLoaded", async () => {
  const stored = await chrome.storage.local.get([STORAGE_KEYS.recentChats]);
  chatInput.value = stored[STORAGE_KEYS.recentChats] || "";
});

saveButton.addEventListener("click", async () => {
  const value = chatInput.value.trim();
  await chrome.storage.local.set({
    [STORAGE_KEYS.recentChats]: value
  });
  statusMessage.textContent = "Saved. New tabs will now use this context.";
});

clearButton.addEventListener("click", async () => {
  chatInput.value = "";
  await chrome.storage.local.set({
    [STORAGE_KEYS.recentChats]: ""
  });
  statusMessage.textContent = "Cleared. Add new chats anytime.";
});
