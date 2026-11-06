import { toCSV, validateRun } from "./data.js";

const $ = (id) => document.getElementById(id);
const format = new Intl.NumberFormat("en");
const pageSize = 18;
const totalCoordinates = 2 * 16_384 + 40 * 258 * 2_048;
const categoryRank = { persona: 0, changed: 1, off_topic: 2, normal: 3, unlabeled: 4, incoherent: 5 };
const categories = {
  persona: "Persona",
  changed: "Changed assistant",
  off_topic: "Off-topic",
  normal: "Normal",
  incoherent: "Incoherent",
  unlabeled: "Unlabeled",
};
const state = {
  run: null,
  byUnit: new Map(),
  filtered: [],
  direction: "both",
  atlasDirection: "up",
  selectedLayer: 0,
  visible: pageSize,
  reply: null,
};
const lookup = $("lookup-dialog");
const reply = $("reply-dialog");
const atlasButtons = [];

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function number(value) {
  return format.format(value);
}

function layerName(layer) {
  return "L" + String(layer).padStart(2, "0");
}

function unitKey(record) {
  return [record.layer, record.expert_type, record.expert ?? "", record.neuron].join(":");
}

function coordinate(record) {
  const expert = record.expert_type === "dense"
    ? "Dense"
    : record.expert_type[0].toUpperCase() + record.expert_type.slice(1) + " " + record.expert;
  return layerName(record.layer) + " · " + expert + " · N" + record.neuron;
}

function directionLabel(direction) {
  return direction === "up" ? "↑ Up" : "↓ Down";
}

function categoryLabel(record) {
  return categories[record.category] ?? categories.unlabeled;
}

function setSegment(container, direction) {
  for (const button of container.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.direction === direction));
  }
}

function attachSegment(id, change) {
  $(id).addEventListener("click", (event) => {
    const button = event.target.closest("button[data-direction]");
    if (!button) return;
    setSegment($(id), button.dataset.direction);
    change(button.dataset.direction);
  });
}

function compareCoordinates(a, b) {
  return a.layer - b.layer
    || a.expert_type.localeCompare(b.expert_type)
    || (a.expert ?? 0) - (b.expert ?? 0)
    || a.neuron - b.neuron
    || Number(b.direction === "up") - Number(a.direction === "up");
}

function compareWeirdness(a, b) {
  return (b.weirdness ?? -1) - (a.weirdness ?? -1) || compareCoordinates(a, b);
}

function compareHighlights(a, b) {
  return categoryRank[a.category ?? "unlabeled"] - categoryRank[b.category ?? "unlabeled"]
    || compareWeirdness(a, b);
}

function sortHighlights(records) {
  const seen = new Map();
  const first = [], variants = [], repeats = [], ordinary = [];
  for (const record of records.sort(compareHighlights)) {
    const category = record.category ?? "unlabeled";
    const text = categoryRank[category] < categoryRank.normal ? record.reply.trim().replace(/\s+/g, " ") : "";
    if (!text) {
      ordinary.push(record);
      continue;
    }
    if (!seen.has(category)) seen.set(category, { names: new Set(), replies: new Set() });
    const group = seen.get(category);
    if (group.replies.has(text)) {
      repeats.push(record);
      continue;
    }
    group.replies.add(text);

    const name = record.name?.trim().replace(/\s+/g, " ").toLowerCase();
    const bucket = name && group.names.has(name) ? variants : first;
    bucket.push(record);
    if (name) group.names.add(name);
  }
  return first.concat(variants, repeats, ordinary);
}

const librarySorts = { coordinate: compareCoordinates, weirdness: compareWeirdness };

function updateLibrary(reset = true) {
  if (reset) state.visible = pageSize;
  const search = $("search").value.trim().toLowerCase();
  const layer = $("layer-filter").value;
  const category = $("category").value;
  const hideSensitive = $("hide-sensitive").checked;

  state.filtered = (state.run?.results ?? []).filter((record) => {
    if (hideSensitive && record.sensitive) return false;
    if (state.direction !== "both" && record.direction !== state.direction) return false;
    if (layer !== "" && record.layer !== Number(layer)) return false;
    if (category && (record.category ?? "unlabeled") !== category) return false;
    return !search || [record.name, record.description, record.reply, coordinate(record), ...(record.tags ?? [])]
      .some((value) => typeof value === "string" && value.toLowerCase().includes(search));
  });
  const sort = $("sort").value;
  if (sort === "highlights") state.filtered = sortHighlights(state.filtered);
  else state.filtered.sort(librarySorts[sort]);

  const visible = state.filtered.slice(0, state.visible);
  $("results").replaceChildren(...visible.map(resultCard));
  const count = state.filtered.length;
  $("result-count").textContent = number(count) + (count === 1 ? " observation" : " observations");
  $("empty").hidden = count !== 0;
  $("export").disabled = count === 0;
  $("more").hidden = visible.length >= count;
  const emptyCopy = !state.run
    ? ["The first run is pending", "Import a run from the experiment runner to browse actual Inkling-Small replies."]
    : state.run.results.length === 0
      ? ["Baseline only", "This run contains the unsteered reply. Import a completed run to compare interventions."]
      : ["No matching observations", "Clear your filters or try a different layer or direction."];
  $("empty-title").textContent = emptyCopy[0];
  $("empty-description").textContent = emptyCopy[1];
  $("empty-action").hidden = Boolean(state.run);
}

function resultCard(record) {
  const card = element("button", "card");
  card.type = "button";
  const excerpt = record.sensitive ? "Sensitive reply. Open to reveal." : (record.reply || "Empty reply");
  card.append(
    element("span", "card-coord", coordinate(record) + " · " + directionLabel(record.direction)),
    element("strong", "card-title", record.name || categoryLabel(record)),
    element("span", "card-excerpt", excerpt),
    element("span", "card-category", categoryLabel(record) + " / " + record.strength + "×"),
  );
  card.addEventListener("click", () => openReply(record));
  return card;
}

function initializeLayers() {
  for (let layer = 0; layer < 42; layer += 1) {
    const option = element("option", "", layerName(layer));
    option.value = String(layer);
    $("layer-filter").append(option);

    const button = element("button", layer < 2 ? "layer-tile dense" : "layer-tile");
    button.type = "button";
    button.append(element("span", "", layerName(layer)), element("small"));
    button.addEventListener("click", () => {
      state.selectedLayer = layer;
      renderAtlas();
    });
    atlasButtons.push(button);
    $("layer-grid").append(button);
  }
}

function renderAtlas() {
  const layerCounts = Array.from({ length: 42 }, () => ({ total: 0, persona: 0 }));
  for (const record of state.run?.results ?? []) {
    if (record.direction !== state.atlasDirection) continue;
    layerCounts[record.layer].total += 1;
    if (record.category === "persona") layerCounts[record.layer].persona += 1;
  }
  atlasButtons.forEach((button, layer) => {
    const { total } = layerCounts[layer];
    button.querySelector("small").textContent = total ? number(total) : "";
    button.classList.toggle("observed", total > 0);
    button.setAttribute("aria-pressed", String(layer === state.selectedLayer));
    button.setAttribute("aria-label", layerName(layer) + (layer < 2 ? ", dense" : ", mixture of experts") + ", " + number(total) + " observations");
  });

  const layer = state.selectedLayer;
  const { total, persona } = layerCounts[layer];
  const description = layer < 2
    ? "Dense MLP · 16,384 neuron coordinates"
    : "256 routed experts + 2 shared experts · 2,048 neurons per expert";
  const observations = number(total) + " " + state.atlasDirection + (total === 1 ? " observation" : " observations");
  const detail = element("div");
  detail.append(
    element("strong", "", layerName(layer) + " · " + description),
    element("p", "", observations + (persona ? " · " + number(persona) + " labeled persona" : "")),
  );
  const filter = element("button", "button button-small", "View in library ↗");
  filter.type = "button";
  filter.addEventListener("click", () => {
    $("layer-filter").value = String(layer);
    $("category").value = "";
    $("search").value = "";
    state.direction = state.atlasDirection;
    setSegment($("library-direction"), state.direction);
    updateLibrary();
    window.location.hash = "library";
  });
  $("layer-detail").replaceChildren(detail, filter);
  const layerReplies = $("layer-replies");
  layerReplies.hidden = total === 0;
  if (!total) {
    layerReplies.replaceChildren();
    return;
  }
  const points = element("div", "observation-points");
  for (const record of (state.run?.results ?? [])
    .filter((record) => record.layer === layer && record.direction === state.atlasDirection)
    .sort(compareCoordinates).slice(0, 96)) {
    const point = element("button", "observation-point");
    point.type = "button";
    point.dataset.category = record.category ?? "unlabeled";
    point.title = coordinate(record) + " · " + categoryLabel(record);
    point.setAttribute("aria-label", point.title);
    point.addEventListener("click", () => openReply(record));
    points.append(point);
  }
  layerReplies.replaceChildren(
    element("p", "", "Recorded neurons" + (total > 96 ? " · first 96" : "")),
    points,
  );
}

function updateLookup() {
  const dense = Number($("lookup-layer").value) < 2;
  $("lookup-neuron").max = dense ? "16383" : "2047";
  $("lookup-type-field").hidden = dense;
  $("lookup-expert-field").hidden = dense;
  $("lookup-type").disabled = dense;
  $("lookup-expert").disabled = dense;
  $("lookup-expert").max = $("lookup-type").value === "routed" ? "255" : "1";
  $("lookup-message").textContent = "";
}

function openReply(record) {
  if (lookup.open) lookup.close();
  state.reply = record;
  const options = state.byUnit.get(unitKey(record)) ?? [];
  const directionButtons = [...options].sort(compareCoordinates).map((option) => {
    const button = element("button", "", directionLabel(option.direction));
    button.type = "button";
    button.setAttribute("aria-pressed", String(option.direction === record.direction));
    button.addEventListener("click", () => openReply(option));
    return button;
  });
  $("reply-options").replaceChildren(...directionButtons);
  $("reply-title").textContent = record.name || categoryLabel(record);
  $("reply-coordinate").textContent = coordinate(record) + " · " + record.strength + "×";
  const scores = [];
  if (record.weirdness != null) scores.push("Weirdness " + record.weirdness + "/10");
  if (record.coherence != null) scores.push("Coherence " + record.coherence + "/10");
  $("reply-label").textContent = [categoryLabel(record), ...scores, ...(record.tags ?? [])].join(" · ");
  $("reply-description").textContent = record.description ?? "";
  $("reply-baseline").textContent = state.run.baseline.reply || "Empty reply";
  $("reveal-reply").hidden = !record.sensitive;
  renderReplyText(record, !record.sensitive);
  if (!reply.open) reply.showModal();
}

function renderReplyText(record, revealed) {
  $("reply-text").hidden = !revealed;
  $("reply-text").textContent = revealed ? (record.reply || "Empty reply") : "";
  $("reply-attempts").hidden = !revealed || (record.attempts?.length ?? 0) <= 1;
  $("reply-attempts").open = false;
  const attemptItems = revealed ? (record.attempts ?? []).map((attempt) => {
    const item = element("div");
    const label = attempt.category ? " · " + categoryLabel(attempt) : "";
    item.append(element("strong", "", attempt.strength + "×" + label), element("pre", "", attempt.reply || "No assistant text."));
    if (attempt.raw_reply && attempt.raw_reply !== attempt.reply) {
      const raw = element("details");
      raw.append(element("summary", "", "Raw output"), element("pre", "", attempt.raw_reply));
      item.append(raw);
    }
    return item;
  }) : [];
  $("attempt-list").replaceChildren(...attemptItems);
}

function loadRun(input) {
  const run = validateRun(input);
  state.run = run;
  state.byUnit = new Map();
  for (const record of run.results) {
    const key = unitKey(record);
    if (!state.byUnit.has(key)) state.byUnit.set(key, []);
    state.byUnit.get(key).push(record);
  }
  const highlight = run.results.filter((record) => !record.sensitive && record.reply.trim()
    && categoryRank[record.category] < categoryRank.normal).sort(compareHighlights)[0];
  $("featured").hidden = !highlight;
  $("featured-reply").replaceChildren(...(highlight ? [resultCard(highlight)] : []));
  const measuredLayers = new Set(run.results.map((record) => record.layer)).size;
  $("atlas-coverage").textContent = `${number(state.byUnit.size)} of ${number(totalCoordinates)} neuron coordinates observed across ${measuredLayers} of 42 layers.`;
  $("baseline").textContent = run.baseline.reply || "Empty reply";
  $("run-prompt").textContent = "“" + run.prompt + "”";
  $("run-status").textContent = run.checkpoint + " · " + number(run.results.length) + " OBSERVATIONS";
  $("run-status").title = "Checkpoint revision " + run.revision;
  $("import-error").hidden = true;
  $("announcement").textContent = "Run loaded. " + number(run.results.length) + " observations.";
  updateLibrary();
  renderAtlas();
}

let importCount = 0;
for (const button of document.querySelectorAll(".upload-trigger")) {
  button.addEventListener("click", () => $("run-file").click());
}
$("run-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const current = ++importCount;
  event.target.value = "";
  try {
    const input = JSON.parse(await file.text());
    if (current === importCount) loadRun(input);
  } catch (error) {
    if (current !== importCount) return;
    $("import-error").textContent = "Could not import this run. " + error.message;
    $("import-error").hidden = false;
    $("import-error").scrollIntoView({ block: "nearest" });
  }
});

$("export").addEventListener("click", () => {
  const blob = new Blob([toCSV(state.filtered)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const download = element("a");
  download.href = url;
  download.download = "inkling-persona-observations.csv";
  download.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("search").addEventListener("input", () => updateLibrary());
for (const id of ["category", "layer-filter", "sort", "hide-sensitive"]) {
  $(id).addEventListener("change", () => updateLibrary());
}
$("more").addEventListener("click", () => {
  state.visible += pageSize;
  updateLibrary(false);
});
attachSegment("library-direction", (direction) => {
  state.direction = direction;
  updateLibrary();
});
attachSegment("atlas-direction", (direction) => {
  state.atlasDirection = direction;
  renderAtlas();
});

$("lookup-open").addEventListener("click", () => {
  $("lookup-layer").value = $("layer-filter").value || String(state.selectedLayer);
  $("lookup-neuron").value = "0";
  $("lookup-expert").value = "0";
  updateLookup();
  lookup.showModal();
});
for (const id of ["lookup-layer", "lookup-type"]) $(id).addEventListener("input", updateLookup);
$("lookup-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const layer = Number($("lookup-layer").value);
  const key = unitKey({
    layer,
    expert_type: layer < 2 ? "dense" : $("lookup-type").value,
    expert: layer < 2 ? null : Number($("lookup-expert").value),
    neuron: Number($("lookup-neuron").value),
  });
  const records = state.byUnit.get(key);
  if (records?.length) {
    openReply(records.find((record) => record.direction === "up") ?? records[0]);
  } else {
    $("lookup-message").textContent = state.run
      ? "This run has no reply for that coordinate."
      : "Import a run to look up replies.";
  }
});
for (const dialog of [lookup, reply]) {
  dialog.querySelector(".dialog-close").addEventListener("click", () => dialog.close());
}
$("reveal-reply").addEventListener("click", () => {
  $("reveal-reply").hidden = true;
  renderReplyText(state.reply, true);
});

initializeLayers();
renderAtlas();
updateLibrary();

try {
  const response = await fetch("./run.json");
  if (response.ok && !importCount) {
    const input = await response.json();
    if (!importCount && input !== null) loadRun(input);
  }
} catch {
  // A bundled run is optional. Browser-local file import remains available.
}
