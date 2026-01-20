const STOP_WORDS = new Set([
  "the",
  "and",
  "that",
  "with",
  "this",
  "from",
  "your",
  "about",
  "into",
  "have",
  "will",
  "just",
  "what",
  "when",
  "where",
  "their",
  "they",
  "them",
  "then",
  "than",
  "does",
  "did",
  "you",
  "are",
  "for",
  "but",
  "not",
  "was",
  "were",
  "can",
  "could",
  "should",
  "would",
  "how",
  "why",
  "who",
  "whom",
  "our",
  "out",
  "off",
  "its",
  "let",
  "get",
  "got",
  "has",
  "had",
  "over",
  "under",
  "more",
  "most",
  "some",
  "many",
  "very",
  "also",
  "like",
  "want",
  "need",
  "help",
  "use",
  "using",
  "make",
  "made",
  "doing",
  "done",
  "work",
  "plan",
  "plans",
  "task",
  "tasks",
  "chat",
  "prompt",
  "model",
  "llm"
]);

const STORAGE_KEYS = {
  recentChats: "recentChats",
  lastServedQuoteId: "lastServedQuoteId"
};

const quoteTextEl = document.getElementById("quoteText");
const quoteAuthorEl = document.getElementById("quoteAuthor");
const quoteTopicEl = document.getElementById("quoteTopic");
const refreshBtn = document.getElementById("refreshQuote");

refreshBtn.addEventListener("click", () => {
  renderQuote();
});

document.addEventListener("DOMContentLoaded", () => {
  renderQuote();
});

async function renderQuote() {
  try {
    const quotes = await loadQuotes();
    const stored = await chrome.storage.local.get([
      STORAGE_KEYS.recentChats,
      STORAGE_KEYS.lastServedQuoteId
    ]);
    const recentChats = stored[STORAGE_KEYS.recentChats] || "";
    const lastServedQuoteId = stored[STORAGE_KEYS.lastServedQuoteId] || "";
    const topics = extractTopics(recentChats);
    const chosen = pickQuote(quotes, topics, lastServedQuoteId);

    quoteTextEl.textContent = `"${chosen.text}"`;
    quoteAuthorEl.textContent = `— ${chosen.author}`;
    quoteTopicEl.textContent = topics.length
      ? `Topic match: ${topics.slice(0, 3).join(", ")}`
      : "Add recent chats to personalize your quotes.";

    await chrome.storage.local.set({
      [STORAGE_KEYS.lastServedQuoteId]: chosen.id
    });
  } catch (error) {
    quoteTextEl.textContent = "Could not load a quote yet.";
    quoteAuthorEl.textContent = "";
    quoteTopicEl.textContent = "Open the options page to add recent chats.";
    console.error(error);
  }
}

async function loadQuotes() {
  const response = await fetch("quotes.json");
  if (!response.ok) {
    throw new Error("Failed to load quotes.");
  }
  return response.json();
}

function extractTopics(text) {
  if (!text) {
    return [];
  }

  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 3 && !STOP_WORDS.has(word));

  const counts = new Map();
  for (const word of words) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }

  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([word]) => word);
}

function pickQuote(quotes, topics, lastServedQuoteId) {
  const topicSet = new Set(topics);
  const matching = quotes.filter((quote) =>
    quote.tags.some((tag) => topicSet.has(tag))
  );

  const candidates = (matching.length ? matching : quotes).filter(
    (quote) => quote.id !== lastServedQuoteId
  );

  const pool = candidates.length ? candidates : quotes;
  return pool[Math.floor(Math.random() * pool.length)];
}
