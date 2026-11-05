const MODEL = "thinkingmachines/Inkling-Small";
const CATEGORIES = ["persona", "changed", "off_topic", "normal", "incoherent"];
const CSV_FIELDS = [
  "layer", "expert_type", "expert", "neuron", "direction", "strength",
  "category", "name", "description", "tags", "weirdness", "coherence", "sensitive", "reply",
];

function requireValue(condition, field, requirement) {
  if (!condition) throw new Error(`${field} ${requirement}.`);
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isIndex(value, limit) {
  return Number.isInteger(value) && value >= 0 && value < limit;
}

function validateReply(value, field) {
  requireValue(isObject(value), field, "must be an object");
  requireValue(typeof value.reply === "string", `${field}.reply`, "must be text");
  requireValue(
    value.raw_reply == null || typeof value.raw_reply === "string",
    `${field}.raw_reply`, "must be text",
  );
  requireValue(
    value.token_ids == null || (Array.isArray(value.token_ids) &&
      value.token_ids.every((token) => Number.isSafeInteger(token) && token >= 0)),
    `${field}.token_ids`, "must contain only nonnegative integer token IDs",
  );
  requireValue(
    value.stop_reason == null || typeof value.stop_reason === "string",
    `${field}.stop_reason`, "must be text",
  );
}

function validateStrength(value, field) {
  requireValue(Number.isFinite(value) && value > 0, field, "must be a positive number");
}

function normalizeResult(result, index, coordinates) {
  const field = `Result ${index + 1}`;
  validateReply(result, field);

  const { layer, expert_type: kind, expert, neuron, direction, strength } = result;
  requireValue(isIndex(layer, 42), `${field}.layer`, "must be an integer from 0 to 41");
  if (layer < 2) {
    requireValue(kind === "dense", `${field}.expert_type`, "must be dense for layers 0 and 1");
    requireValue(expert == null, `${field}.expert`, "must be null for a dense layer");
    requireValue(isIndex(neuron, 16384), `${field}.neuron`, "must be an integer from 0 to 16,383");
  } else {
    requireValue(
      kind === "shared" || kind === "routed",
      `${field}.expert_type`, "must be shared or routed for layers 2 through 41",
    );
    const expertCount = kind === "shared" ? 2 : 256;
    requireValue(
      isIndex(expert, expertCount), `${field}.expert`,
      `must be an integer from 0 to ${expertCount - 1}`,
    );
    requireValue(isIndex(neuron, 2048), `${field}.neuron`, "must be an integer from 0 to 2,047");
  }
  requireValue(direction === "up" || direction === "down", `${field}.direction`, "must be up or down");
  validateStrength(strength, `${field}.strength`);

  const key = `${layer}:${kind}:${expert ?? ""}:${neuron}:${direction}`;
  requireValue(!coordinates.has(key), field, "duplicates an earlier coordinate and direction");
  coordinates.add(key);

  const category = result.category === "unlabeled" ? null : result.category ?? null;
  requireValue(category === null || CATEGORIES.includes(category), `${field}.category`, "is unknown");
  const { name = null, description = null, weirdness = null, coherence = null, sensitive = null } = result;
  for (const [label, value] of Object.entries({ name, description })) {
    requireValue(value === null || typeof value === "string", `${field}.${label}`, "must be text");
  }
  for (const [label, value] of Object.entries({ weirdness, coherence })) {
    requireValue(
      value === null || (Number.isFinite(value) && value >= 0 && value <= 10),
      `${field}.${label}`, "must be a number from 0 to 10",
    );
  }
  requireValue(
    sensitive === null || typeof sensitive === "boolean",
    `${field}.sensitive`, "must be true, false, or null",
  );
  const tags = result.tags ?? [];
  requireValue(
    Array.isArray(tags) && tags.every((tag) => typeof tag === "string"),
    `${field}.tags`, "must be a list of text tags",
  );
  const attempts = result.attempts ?? [];
  requireValue(Array.isArray(attempts), `${field}.attempts`, "must be a list");
  for (const [attemptIndex, attempt] of attempts.entries()) {
    const attemptField = `${field}, attempt ${attemptIndex + 1}`;
    validateReply(attempt, attemptField);
    validateStrength(attempt.strength, `${attemptField}.strength`);
  }

  return {
    ...result, expert: expert ?? null, category, name, description,
    tags, weirdness, coherence, sensitive, attempts,
  };
}

export function validateRun(value) {
  requireValue(isObject(value), "Run", "must be a JSON object");
  requireValue(value.model === MODEL, "Run model", `must be ${MODEL}`);
  for (const field of ["checkpoint", "revision", "prompt"]) {
    const entry = value[field];
    requireValue(typeof entry === "string" && entry.trim() !== "", `Run ${field}`, "must be nonempty text");
  }
  validateReply(value.baseline, "Run baseline");
  requireValue(Array.isArray(value.results), "Run results", "must be a list");
  const coordinates = new Set();
  return {
    ...value,
    results: value.results.map((result, index) => normalizeResult(result, index, coordinates)),
  };
}

function csvCell(value) {
  let cell = String(value ?? "");
  // A visible apostrophe stays ordinary text if a spreadsheet saves the CSV again.
  if (typeof value === "string" && /^[\p{White_Space}\p{Cf}\p{Cc}]*[=+\-@＝＋－＠\t\r\n]/u.test(cell)) {
    cell = `’${cell}`;
  }
  return `"${cell.replaceAll('"', '""')}"`;
}

export function toCSV(results) {
  const lines = [CSV_FIELDS.join(",")];
  for (const result of results) {
    const cells = CSV_FIELDS.map((field) => {
      const value = field === "tags" ? result.tags?.join(" | ") : result[field];
      return csvCell(value);
    });
    lines.push(cells.join(","));
  }
  return `\uFEFF${lines.join("\r\n")}\r\n`;
}
