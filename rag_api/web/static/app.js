"use strict";

/*
 * E-commerce RAG console. Plain JS, no dependencies.
 * Security: untrusted text (questions, product data, LLM answers) is inserted with
 * textContent, or HTML-escaped BEFORE the minimal markdown formatting is applied.
 */

const CHAT_TIMEOUT_MS = 120_000;
const HEALTH_INTERVAL_MS = 30_000;
const CATALOG_INTERVAL_MS = 5_000;

const $ = (sel) => document.querySelector(sel);

/* ---------- tiny DOM helper ---------- */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value; // only ever used with escaped/rendered markdown
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

/* ---------- HTTP ---------- */
async function api(path, { method = "GET", body, timeout = 15_000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    let data = null;
    try { data = await response.json(); } catch { /* non-JSON body */ }
    return { ok: response.ok, status: response.status, data };
  } catch (err) {
    const message = err.name === "AbortError" ? "Request timed out" : "Cannot reach the API server";
    return { ok: false, status: 0, data: { detail: message } };
  } finally {
    clearTimeout(timer);
  }
}

function errorMessage(result) {
  const detail = result.data && result.data.detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors (422)
    return detail.map((d) => `${(d.loc || []).slice(1).join(".") || "request"}: ${d.msg}`).join(" · ");
  }
  if (typeof detail === "string") return detail;
  return `HTTP ${result.status}`;
}

/* ---------- minimal, safe markdown ---------- */
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inlineMarkdown(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>");
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  let html = "";
  let open = null; // "ul" | "ol"
  const close = () => { if (open) { html += `</${open}>`; open = null; } };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    let m;
    if (!line.trim()) { close(); continue; }
    if ((m = line.match(/^(\s*)[*-]\s+(.*)$/))) {
      if (open !== "ul") { close(); html += "<ul>"; open = "ul"; }
      html += `<li${m[1].length >= 2 ? ' class="sub"' : ""}>${inlineMarkdown(m[2])}</li>`;
    } else if ((m = line.match(/^\s*(\d+)[.)]\s+(.*)$/))) {
      if (open !== "ol") { close(); html += `<ol start="${Number(m[1])}">`; open = "ol"; }
      html += `<li>${inlineMarkdown(m[2])}</li>`;
    } else if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
      close(); html += `<h4>${inlineMarkdown(m[1])}</h4>`;
    } else {
      close(); html += `<p>${inlineMarkdown(line)}</p>`;
    }
  }
  close();
  return html;
}

/* ---------- formatting ---------- */
function formatPrice(value, currency) {
  const n = Number(value);
  const formatted = Number.isFinite(n)
    ? n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : String(value);
  return `${formatted} ${currency || ""}`.trim();
}

function stockCell(quantity) {
  return quantity > 0
    ? el("span", { class: "stock-in", text: String(quantity) })
    : el("span", { class: "stock-out", text: "out" });
}

/* ---------- health ---------- */
async function refreshHealth() {
  const result = await api("/health", { timeout: 20_000 });
  const deps = (result.data && result.data.dependencies) || {};
  document.querySelectorAll(".pill[data-dep]").forEach((pill) => {
    const dep = deps[pill.dataset.dep];
    pill.classList.remove("ok", "down");
    if (!dep) {
      pill.title = result.status === 0 ? errorMessage(result) : "unknown";
      return;
    }
    pill.classList.add(dep.status === "ok" ? "ok" : "down");
    pill.title = dep.status === "ok" ? "OK" : (dep.error || "unavailable");
  });
}

/* ---------- chat ---------- */
const form = $("#ask-form");
const questionInput = $("#question");
const askButton = $("#ask-btn");
const conversation = $("#conversation");

function productsTable(products) {
  const rows = products.map((p) => {
    const fill = el("div", { class: "score-fill" });
    fill.style.width = `${Math.max(0, Math.min(1, p.score)) * 100}%`; // CSSOM: allowed by CSP
    return el("tr", {}, [
      el("td", {}, [
        el("div", { class: "product-name", text: p.name }),
        el("div", { class: "product-sub", text: [p.brand, p.category].filter(Boolean).join(" · ") }),
      ]),
      el("td", { class: "num", text: formatPrice(p.price, p.currency) }),
      el("td", { class: "num" }, stockCell(p.stock_quantity)),
      el("td", {}, el("div", { class: "score", title: "Semantic similarity (cosine)" }, [
        el("div", { class: "score-bar" }, fill),
        el("span", { class: "score-value", text: p.score.toFixed(2) }),
      ])),
    ]);
  });
  return el("table", {}, [
    el("thead", {}, el("tr", {}, [
      el("th", { text: "Product" }), el("th", { class: "num", text: "Price" }),
      el("th", { class: "num", text: "Stock" }), el("th", { text: "Relevance" }),
    ])),
    el("tbody", {}, rows),
  ]);
}

function renderTurn(question) {
  const meta = el("span", { class: "turn-meta", text: "" });
  const body = el("div", { class: "turn-body" },
    el("div", { class: "loading" }, [el("div", { class: "spinner" }), "Searching products and generating an answer…"]));
  const turn = el("article", { class: "turn" }, [
    el("div", { class: "turn-question" }, [el("span", { class: "q", dir: "auto", text: question }), meta]),
    body,
  ]);
  $("#empty-state")?.remove();
  conversation.prepend(turn);
  return { body, meta };
}

function renderAnswer(body, data) {
  body.replaceChildren();

  const filters = el("div", { class: "filters" });
  if (data.applied_filters.length) {
    data.applied_filters.forEach((f) =>
      filters.append(el("span", { class: "badge exact", title: "Enforced exactly by PostgreSQL", text: `✓ ${f}` })));
  } else {
    filters.append(el("span", {
      class: "badge semantic",
      title: "No price/stock constraint detected: products were matched by meaning only",
      text: "semantic match only",
    }));
  }
  body.append(filters);
  body.append(el("div", { class: "answer", dir: "auto", html: renderMarkdown(data.answer) }));

  if (data.products.length) {
    const details = el("details", { class: "sources" }, [
      el("summary", { text: `Sources: ${data.products.length} product(s) from PostgreSQL` }),
      productsTable(data.products),
    ]);
    details.open = true;
    body.append(details);
  }
}

async function ask(question) {
  question = question.trim();
  if (!question) return;

  askButton.disabled = true;
  const { body, meta } = renderTurn(question);
  const started = performance.now();

  const result = await api("/chat", {
    method: "POST",
    body: { question, top_k: Number($("#top-k").value) },
    timeout: CHAT_TIMEOUT_MS,
  });

  meta.textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
  if (result.ok) {
    renderAnswer(body, result.data);
  } else {
    body.replaceChildren(el("div", { class: "error-box", text: `Error: ${errorMessage(result)}` }));
    refreshHealth();
  }
  askButton.disabled = false;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(questionInput.value);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  $("#char-count").textContent = `${questionInput.value.length} / 500`;
});

document.querySelectorAll(".chip[data-q]").forEach((chip) =>
  chip.addEventListener("click", () => {
    questionInput.value = chip.dataset.q;
    questionInput.dispatchEvent(new Event("input"));
    form.requestSubmit();
  }));

$("#clear-btn").addEventListener("click", () => {
  conversation.replaceChildren();
  questionInput.value = "";
  questionInput.dispatchEvent(new Event("input"));
  questionInput.focus();
});

/* ---------- live catalog ---------- */
const STATUS_LABELS = { synced: "synced", stale: "stale", missing: "not indexed", unknown: "unknown" };
let previousCatalog = new Map(); // id -> "price|stock|active|status", to flash changed rows
let catalogTimer = null;

function countCard(value, label, mismatch = false) {
  return el("div", { class: `count-card${mismatch ? " mismatch" : ""}` }, [
    el("div", { class: "n", text: value }),
    el("div", { class: "l", text: label }),
  ]);
}

async function refreshCatalog() {
  const result = await api("/catalog?limit=200");
  const errorBox = $("#catalog-error");

  if (!result.ok) {
    errorBox.textContent = `Catalog unavailable: ${errorMessage(result)}`;
    errorBox.classList.remove("hidden");
    return;
  }
  const data = result.data;
  if (data.qdrant_error) {
    errorBox.textContent = `${data.qdrant_error}: showing PostgreSQL only.`;
    errorBox.classList.remove("hidden");
  } else {
    errorBox.classList.add("hidden");
  }

  const synced = data.items.filter((i) => i.sync_status === "synced").length;
  const active = data.items.filter((i) => i.is_active).length;
  $("#counts").replaceChildren(
    countCard(data.postgres_count, "in PostgreSQL"),
    countCard(data.qdrant_count ?? "?", "points in Qdrant", data.qdrant_count !== data.postgres_count),
    countCard(`${synced}/${data.items.length}`, "in sync", synced !== data.items.length),
    countCard(active, "active"),
  );

  const next = new Map();
  const rows = data.items.map((item) => {
    const signature = [item.price, item.stock_quantity, item.is_active, item.sync_status].join("|");
    next.set(item.id, signature);
    const changed = previousCatalog.size > 0 && previousCatalog.get(item.id) !== signature;

    const name = el("div", { class: "product-name", text: item.name });
    if (!item.is_active) name.append(el("span", { class: "tag", text: "inactive" }));

    const statusTitle = item.sync_status === "stale"
      ? `Index differs from PostgreSQL on: ${item.differences.join(", ")}`
      : STATUS_LABELS[item.sync_status];

    return el("tr", { class: [item.is_active ? "" : "inactive", changed ? "flash" : ""].join(" ").trim() || null }, [
      el("td", {}, [name, el("div", { class: "product-sub", text: [item.brand, item.category].filter(Boolean).join(" · ") })]),
      el("td", { class: "num", text: formatPrice(item.price, item.currency) }),
      el("td", { class: "num" }, stockCell(item.stock_quantity)),
      el("td", { title: statusTitle }, el("span", { class: "status-label" }, [
        el("span", { class: `status ${item.sync_status}` }),
        STATUS_LABELS[item.sync_status],
      ])),
    ]);
  });

  $("#catalog-body").replaceChildren(...rows);
  previousCatalog = next;
}

$("#catalog-refresh").addEventListener("click", refreshCatalog);
$("#auto-refresh").addEventListener("change", (event) => {
  clearInterval(catalogTimer);
  catalogTimer = event.target.checked ? setInterval(refreshCatalog, CATALOG_INTERVAL_MS) : null;
});
$("#health-refresh").addEventListener("click", refreshHealth);

/* ---------- start ---------- */
refreshHealth();
refreshCatalog();
setInterval(refreshHealth, HEALTH_INTERVAL_MS);
questionInput.focus();
