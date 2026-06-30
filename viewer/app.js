console.log("viewer app.js loaded", new Date().toISOString());
window.addEventListener("error", (e) => {
  const el = document.getElementById("empty-state");
  if (el) {
    el.hidden = false;
    el.textContent = `Script error: ${e.message}`;
  }
});

const MANIFEST_URL = "../data/manifest.json";
const ROLLOUTS_MANIFEST_URL = "../eval/var/rollouts_manifest.json";

const VIDEO_EXT = new Set(["mp4", "mov", "webm", "m4v"]);
const AUDIO_EXT = new Set(["wav", "mp3", "m4a", "aac", "flac"]);
const IMAGE_EXT = new Set(["jpg", "jpeg", "png", "gif", "webp"]);
const PDF_EXT = new Set(["pdf"]);
const TEXT_EXT = new Set(["txt", "md", "json", "csv", "srt", "vtt", "log"]);

// task_folder -> [policy records]; populated at startup.
let ROLLOUTS = {};

function ext(name) {
  const parts = name.split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

function fmtBytes(bytes) {
  if (bytes == null) return "unknown size";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(1)} ${units[i]}`;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

function renderFileCard(file) {
  const path = "../" + file.path;
  const e = ext(file.name);
  const card = el("div", { class: "file-card" });
  card.appendChild(el("div", { class: "file-name", text: file.name }));
  card.appendChild(
    el("div", { class: "file-meta", text: `${fmtBytes(file.size_bytes)} • .${e || "?"}` })
  );

  if (VIDEO_EXT.has(e)) {
    card.appendChild(el("video", { src: path, controls: "", preload: "metadata" }));
  } else if (AUDIO_EXT.has(e)) {
    card.appendChild(el("audio", { src: path, controls: "", style: "width:100%" }));
  } else if (IMAGE_EXT.has(e)) {
    card.appendChild(el("img", { src: path }));
  } else if (PDF_EXT.has(e)) {
    card.appendChild(el("iframe", { src: path }));
  } else if (TEXT_EXT.has(e)) {
    const pre = el("pre", { class: "text-preview", text: "Loading…" });
    card.appendChild(pre);
    fetch(path)
      .then((r) => r.text())
      .then((t) => {
        pre.textContent = t;
      })
      .catch((err) => {
        pre.textContent = `Could not load file: ${err}`;
      });
  } else {
    card.appendChild(
      el("div", {}, [
        el("a", {
          class: "download-link",
          href: path,
          download: "",
          text: "↓ Download (no inline preview for this file type)",
        }),
      ])
    );
  }
  return card;
}

function renderFileList(files) {
  const wrap = el("div", {});
  if (!files || files.length === 0) {
    wrap.appendChild(el("div", { class: "no-files", text: "No files." }));
    return wrap;
  }
  for (const f of files) wrap.appendChild(renderFileCard(f));
  return wrap;
}

function renderRubric(rubricItems, prettyText) {
  const wrap = el("div", {});
  const maxScore = rubricItems.reduce((s, r) => s + (r.score || 0), 0);
  const summary = el("div", { class: "rubric-summary" }, [
    el("span", { html: `<b>${rubricItems.length}</b> criteria` }),
    el("span", { html: `<b>${maxScore}</b> max points` }),
  ]);
  wrap.appendChild(summary);

  const toggleBtn = el("button", { class: "toggle-pretty", text: "Show raw rubric text" });
  const prettyBox = el("div", { id: "rubric-pretty-box", hidden: "" });
  prettyBox.textContent = prettyText || "";
  toggleBtn.addEventListener("click", () => {
    const hidden = prettyBox.hasAttribute("hidden");
    if (hidden) {
      prettyBox.removeAttribute("hidden");
      toggleBtn.textContent = "Hide raw rubric text";
    } else {
      prettyBox.setAttribute("hidden", "");
      toggleBtn.textContent = "Show raw rubric text";
    }
  });
  wrap.appendChild(toggleBtn);
  wrap.appendChild(prettyBox);

  const table = el("table", { class: "rubric-table" });
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", { text: "Score" }),
      el("th", { text: "Criterion" }),
      el("th", { text: "Tags" }),
      el("th", { text: "Author" }),
    ]),
  ]);
  table.appendChild(thead);
  const tbody = el("tbody", {});
  for (const item of rubricItems) {
    const tags = (item.tags || []).map((t) => el("span", { class: "tag-pill", text: t }));
    if (item.required) tags.push(el("span", { class: "tag-pill", text: "required" }));
    if (item.read_only) tags.push(el("span", { class: "tag-pill", text: "read_only" }));
    const row = el("tr", {}, [
      el("td", {}, [el("span", { class: "score-pill", text: `+${item.score ?? "?"}` })]),
      el("td", { text: item.criterion || "" }),
      el("td", {}, tags),
      el("td", { text: item.author_type || "" }),
    ]);
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

const STATUS_LABEL = {
  submitted: "submitted",
  no_deliverable: "no deliverable",
  no_submission: "no submission",
};

function renderRollouts(policies) {
  const wrap = el("div", {});
  if (!policies || policies.length === 0) {
    wrap.appendChild(
      el("div", {
        class: "no-files",
        text: "No custom rollouts recorded for this task.",
      })
    );
    return wrap;
  }

  const nSub = policies.filter((p) => p.submitted).length;
  wrap.appendChild(
    el("div", { class: "rubric-summary" }, [
      el("span", { html: `<b>${policies.length}</b> policies` }),
      el("span", { html: `<b>${nSub}</b> submitted` }),
    ])
  );

  for (const p of policies) {
    const card = el("div", { class: "rollout-card" });

    const header = el("div", { class: "rollout-header" }, [
      el("span", { class: "rollout-policy", text: p.policy }),
      el("span", {
        class: `status-pill status-${p.status}`,
        text: STATUS_LABEL[p.status] || p.status,
      }),
    ]);
    if (p.rounds != null) {
      header.appendChild(el("span", { class: "rollout-rounds", text: `${p.rounds} rounds` }));
    }
    card.appendChild(header);

    if (p.notes) {
      card.appendChild(el("div", { class: "rollout-notes", text: p.notes }));
    }

    if (p.steps && p.steps.length) {
      const details = el("details", { class: "rollout-steps" });
      details.appendChild(el("summary", { text: `${p.steps.length} steps` }));
      const ol = el("ol", {});
      for (const s of p.steps) ol.appendChild(el("li", { text: s }));
      details.appendChild(ol);
      card.appendChild(details);
    }

    if (p.deliverable_files && p.deliverable_files.length) {
      card.appendChild(el("div", { class: "rollout-files-label", text: "Deliverable" }));
      card.appendChild(renderFileList(p.deliverable_files));
    } else {
      card.appendChild(
        el("div", { class: "no-files", text: "No deliverable files produced." })
      );
    }

    wrap.appendChild(card);
  }
  return wrap;
}

async function loadTaskExtras(task) {
  const [rubricItems, prettyText] = await Promise.all([
    fetch(`../data/${task.folder}/rubric.json`).then((r) => r.json()),
    fetch(`../data/${task.folder}/rubric_pretty.txt`).then((r) => r.text()),
  ]);
  return { rubricItems, prettyText };
}

function setActiveTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.hidden = p.id !== `tab-${name}`;
  });
}

async function showTask(task) {
  try {
    document.getElementById("empty-state").hidden = true;
    const view = document.getElementById("task-view");
    view.hidden = false;

    document.getElementById("task-occupation").textContent = task.occupation;
    document.getElementById("task-meta").innerHTML = `
      <span>Folder: <b>${task.folder}</b></span>
      <span>Sector: <b>${task.sector}</b></span>
      <span>Task ID: <b>${task.task_id}</b></span>
      <span>Reference files: <b>${task.reference_files.length}</b></span>
      <span>Deliverable files: <b>${task.deliverable_files.length}</b></span>
    `;

    document.getElementById("tab-prompt").textContent = task.prompt;

    const refPanel = document.getElementById("tab-reference");
    refPanel.innerHTML = "";
    refPanel.appendChild(renderFileList(task.reference_files));

    const delivPanel = document.getElementById("tab-deliverable");
    delivPanel.innerHTML = "";
    delivPanel.appendChild(renderFileList(task.deliverable_files));

    const policies = ROLLOUTS[task.folder] || [];
    const rolloutPanel = document.getElementById("tab-rollouts");
    rolloutPanel.innerHTML = "";
    rolloutPanel.appendChild(renderRollouts(policies));
    const rolloutTab = document.querySelector('.tab-btn[data-tab="rollouts"]');
    rolloutTab.textContent = policies.length ? `Rollouts (${policies.length})` : "Rollouts";

    const rubricPanel = document.getElementById("tab-rubric");
    rubricPanel.innerHTML = "Loading rubric…";
    const { rubricItems, prettyText } = await loadTaskExtras(task);
    rubricPanel.innerHTML = "";
    rubricPanel.appendChild(renderRubric(rubricItems, prettyText));
  } catch (err) {
    console.error("showTask failed", err);
    document.getElementById("empty-state").hidden = false;
    document.getElementById("empty-state").textContent = `Error loading task "${task.folder}": ${err}`;
    document.getElementById("task-view").hidden = true;
  }
}

function renderSidebar(tasks) {
  const list = document.getElementById("task-list");
  list.innerHTML = "";
  document.getElementById("task-count").textContent = `${tasks.length} tasks`;

  tasks.forEach((task, i) => {
    const item = el("li", { class: "task-item" }, [
      el("div", { class: "occ", text: task.occupation }),
      el("div", { class: "folder", text: task.folder }),
      el("div", { class: "sector", text: `${task.sector} • ${task.num_rubric_items} rubric items • ${task.rubric_max_score} pts` }),
    ]);
    item.addEventListener("click", () => {
      document.querySelectorAll(".task-item").forEach((n) => n.classList.remove("active"));
      item.classList.add("active");
      setActiveTab("prompt");
      showTask(task);
    });
    list.appendChild(item);
    if (i === 0) {
      item.classList.add("active");
      setActiveTab("prompt");
      showTask(task);
    }
  });
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
});

Promise.all([
  fetch(MANIFEST_URL).then((r) => r.json()),
  // Rollouts are optional — tolerate a missing/empty manifest.
  fetch(ROLLOUTS_MANIFEST_URL)
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({})),
])
  .then(([tasks, rollouts]) => {
    ROLLOUTS = rollouts || {};
    tasks.sort((a, b) => a.occupation.localeCompare(b.occupation) || a.folder.localeCompare(b.folder));
    renderSidebar(tasks);
  })
  .catch((err) => {
    document.getElementById("empty-state").textContent = `Failed to load manifest: ${err}`;
  });
