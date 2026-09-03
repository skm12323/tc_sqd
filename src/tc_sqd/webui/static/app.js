/* tc_sqd WebUI 前端逻辑 (无构建链: 原生 JS + 自绘 SVG 图表)。
 * 流程: 初始化(presets/caps) → 表单收集 → POST /api/run → 轮询 /api/job
 * → 运行中实时轨迹图 → 结束后汇总卡片/四图/seed 表/原始 JSON。 */
"use strict";

const $ = (id) => document.getElementById(id);
const PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
                 "#0891b2", "#db2777", "#65a30d", "#475569", "#ea580c"];
const PHASE_TEXT = {
  queued: "排队中",
  integrals: "构建积分 (SCF)",
  reference: "全空间参考计算",
  run: "计算中",
};
let pollTimer = null;

/* ---------------- API ---------------- */
async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== null) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `${r.status} ${r.statusText}`);
  return j;
}

/* ---------------- 数字格式化 ---------------- */
function fmtNum(v, digits = 10) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (v === 0) return "0";
  if (a >= 1e6 || a < 1e-3) return v.toExponential(3);
  return String(+v.toPrecision(digits));
}
function fmtInt(v) { return (v ?? 0).toLocaleString("en-US"); }

/* ---------------- SVG 图表 ---------------- */
const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
}
function niceTicks(min, max, n = 5) {
  if (!(max > min)) return [min];
  const step0 = (max - min) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step)
    ticks.push(t);
  return ticks;
}

/* 折线图: series = [{name, points: [{x, y}]}]; opts = {xLabel, yLabel, logY} */
function lineChart(container, series, opts = {}) {
  container.innerHTML = "";
  const all = series.flatMap((s) => s.points);
  const box = document.createElement("div");
  box.className = "chart";
  container.appendChild(box);
  if (!all.length) {
    box.innerHTML = "<div class='title'>" + (opts.title || "") + "</div><p class='hint'>无数据</p>";
    return;
  }
  const W = 560, H = 290, L = 70, R = 14, T = 14, B = 40;
  const xs = all.map((p) => p.x);
  const xmin = Math.min(0, ...xs), xmax = Math.max(1, ...xs);
  let ys = all.map((p) => p.y);
  if (opts.logY) ys = ys.filter((y) => y > 0);
  if (!ys.length) ys = [0, 1];
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (opts.logY) {
    ymin = Math.pow(10, Math.floor(Math.log10(ymin)));
    ymax = Math.pow(10, Math.ceil(Math.log10(ymax) + 1e-12));
  } else {
    const pad = (ymax - ymin || Math.abs(ymax) || 1) * 0.08;
    ymin -= pad; ymax += pad;
  }
  const scX = (x) => L + ((x - xmin) / (xmax - xmin || 1)) * (W - L - R);
  const linY = (y) => T + (1 - (y - ymin) / (ymax - ymin || 1)) * (H - T - B);
  const logYf = (y) => {
    const lmin = Math.log10(ymin), lmax = Math.log10(ymax);
    return T + (1 - (Math.log10(y) - lmin) / (lmax - lmin || 1)) * (H - T - B);
  };
  const scY = opts.logY ? logYf : linY;

  const root = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, box);
  const yTicks = opts.logY
    ? (() => {
        const ts = [];
        for (let e = Math.log10(ymin); e <= Math.log10(ymax) + 1e-9; e++) ts.push(Math.pow(10, e));
        return ts.length > 8 ? ts.filter((_, i) => i % 2 === 0) : ts;
      })()
    : niceTicks(ymin, ymax, 5);
  for (const t of yTicks) {
    const y = opts.logY ? logYf(t) : linY(t);
    svgEl("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: "#e8edf4", "stroke-width": 1 }, root);
    svgEl("text", { x: L - 6, y: y + 4, "text-anchor": "end", "font-size": 10.5,
                    fill: "#66707f", "font-family": "ui-monospace, monospace" },
          root).textContent = opts.logY ? t.toExponential(0) : fmtNum(t, 3);
  }
  for (const t of niceTicks(xmin, xmax, 6)) {
    const x = scX(t);
    svgEl("line", { x1: x, x2: x, y1: T, y2: H - B, stroke: "#f0f3f8", "stroke-width": 1 }, root);
    svgEl("text", { x: x, y: H - B + 15, "text-anchor": "middle", "font-size": 10.5,
                    fill: "#66707f" }, root).textContent = String(+t.toPrecision(4));
  }
  svgEl("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "#aab4c3" }, root);
  svgEl("line", { x1: L, x2: L, y1: T, y2: H - B, stroke: "#aab4c3" }, root);
  if (opts.yLabel) svgEl("text", { x: 12, y: T + 4, "font-size": 11, fill: "#66707f" }, root).textContent = opts.yLabel;
  if (opts.xLabel) svgEl("text", { x: W - R, y: H - 6, "text-anchor": "end", "font-size": 11, fill: "#66707f" }, root).textContent = opts.xLabel;

  series.forEach((s, i) => {
    if (s.points.length < 1) return;
    const pts = opts.logY ? s.points.filter((p) => p.y > 0) : s.points;
    if (!pts.length) return;
    const d = pts.map((p, k) => `${k ? "L" : "M"}${scX(p.x).toFixed(1)},${scY(p.y).toFixed(1)}`).join("");
    svgEl("path", { d, fill: "none", stroke: s.color || PALETTE[i % PALETTE.length], "stroke-width": 1.8 }, root);
    for (const p of pts.slice(0, 1).concat(pts.slice(-1)))
      svgEl("circle", { cx: scX(p.x), cy: scY(p.y), r: 2.4,
                        fill: s.color || PALETTE[i % PALETTE.length] }, root);
  });

  const title = document.createElement("div");
  title.className = "title";
  title.textContent = opts.title || "";
  box.appendChild(title);
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = series
    .map((s, i) => `<span class="sw" style="background:${s.color || PALETTE[i % PALETTE.length]}"></span>${s.name}`)
    .join("");
  box.appendChild(legend);
}

/* 柱状图: items = [{label, value}] */
function barChart(container, items, opts = {}) {
  const series = [{ name: opts.title || "", color: opts.color || PALETTE[0],
                    points: items.map((it, i) => ({ x: i, y: it.value })) }];
  lineChart(container, series, { ...opts, xLabel: "seed 序号" });
}

/* ---------------- 表单 ---------------- */
let PRESETS = [];
let LAST_PREVIEW = null;

function applyPreset(p) {
  $("sys_geometry").value = p.geometry;
  $("sys_basis").value = p.basis;
  $("sys_charge").value = p.charge;
  $("sys_spin").value = p.spin;
  $("sys_ncore").value = p.n_core;
  $("sys_nvirt").value = p.n_virtual;
  $("sys_scf").value = p.scf;
  $("preset_desc").textContent = p.desc || "";
  refreshPreview();
}

function currentSystem() {
  return {
    geometry: $("sys_geometry").value,
    basis: $("sys_basis").value.trim() || "sto-3g",
    charge: +$("sys_charge").value || 0,
    spin: +$("sys_spin").value || 0,
    n_core: +$("sys_ncore").value || 0,
    n_virtual: +$("sys_nvirt").value || 0,
    scf: $("sys_scf").value,
  };
}

async function refreshPreview() {
  const box = $("sys_preview");
  box.classList.remove("err");
  box.textContent = "预览中…";
  try {
    const { preview } = await api("/api/preview", "POST", { system: currentSystem() });
    LAST_PREVIEW = preview;
    const lines = [
      `${preview.formula}  ·  活性 (${preview.na}α, ${preview.nb}β) @ ${preview.norb} 轨道` +
        (preview.n_core || preview.n_virtual
          ? `  (n_core=${preview.n_core}, n_virtual=${preview.n_virtual}, 全 ${preview.norb_full} 轨道)` : ""),
      `全空间维度 = ${fmtInt(preview.dim_full)}`,
    ];
    if (preview.spin_resolved) lines.push("UHF 自旋分辨五积分 (active/SCI 路径已支持)");
    box.textContent = lines.join("\n");
    if ((preview.warnings || []).length) {
      box.classList.add("err");
      box.textContent += "\n⚠ " + preview.warnings.join("\n⚠ ");
    }
  } catch (e) {
    LAST_PREVIEW = null;
    box.classList.add("err");
    box.textContent = "⚠ " + e.message;
  }
}

function methodNeedsSampling() {
  return ["sqd_active", "cipsi", "hci"].includes($("method").value);
}

function refreshVisibility() {
  const m = $("method").value;
  document.querySelectorAll(".method-group").forEach((el) => {
    const wantSampling = el.dataset.for === "sampling" && methodNeedsSampling();
    el.style.display = el.dataset.for === m || wantSampling ? "" : "none";
  });
  $("hf_weight_row").style.display = $("sample_mode").value === "hf" ? "" : "none";
  document.querySelectorAll(".multiseed-only").forEach((el) => {
    el.style.display = $("multiseed").checked ? "" : "none";
  });
  const rm = $("ref_mode").value;
  document.querySelectorAll(".ref-limit-only").forEach((el) => {
    el.style.display = rm === "auto" ? "" : "none";
  });
  document.querySelectorAll(".ref-manual-only").forEach((el) => {
    el.style.display = rm === "manual" ? "" : "none";
  });
}

function collectParams() {
  const method = $("method").value;
  const sampling = {
    shots: +$("shots").value || 500,
    seed: +$("seed").value || 0,
    mode: $("sample_mode").value,
    hf_weight: +$("hf_weight").value,
    n_seeds: $("multiseed").checked ? +$("n_seeds").value || 1 : 1,
  };
  const params = { backend: $("backend").value };
  const num = (id, dflt) => { const v = $(id).value.trim(); return v === "" ? dflt : +v; };
  if (method === "sqd_active") {
    Object.assign(params, {
      max_rounds: num("sqd_max_rounds", 10),
      max_strings: num("sqd_max_strings", 0),
      n_active_per_round: num("sqd_n_active", 50),
      dom_thresh: num("sqd_dom_thresh", 1e-3),
      pt2_floor: num("sqd_pt2_floor", 1e-7),
      coverage_closure: $("sqd_coverage_closure").checked,
      warm_start: $("sqd_warm_start").checked,
      tail_suppression: $("sqd_tail_suppression").checked,
      tail_shots_ref: num("sqd_tail_shots_ref", 0),
      prune_keep: num("sqd_prune_keep", 1.0),
      eigsh_tol: num("sqd_eigsh_tol", null),
      energy_tol: num("sqd_energy_tol", null),
    });
  } else if (method === "cipsi") {
    Object.assign(params, {
      max_strings: num("cip_max_strings", 0),
      max_iter: num("cip_max_iter", 40),
      dom_thresh: num("cip_dom_thresh", 1e-3),
      pt2_floor: num("cip_pt2_floor", 1e-7),
    });
  } else if (method === "hci") {
    Object.assign(params, {
      eps_hb: num("hci_eps_hb", 1e-3),
      max_iter: num("hci_max_iter", 40),
      dom_thresh: num("hci_dom_thresh", 1e-3),
      pt2_floor: num("hci_pt2_floor", 1e-7),
      use_seed: $("hci_use_seed").checked,
    });
  }
  const reference = {
    mode: $("ref_mode").value,
    dim_limit: +$("ref_dim_limit").value || 200000,
  };
  if (reference.mode === "manual") reference.value = +$("ref_value").value;
  return { system: currentSystem(), method, sampling, params, reference };
}

/* ---------------- 运行与轮询 ---------------- */
async function submitJob() {
  const body = collectParams();
  if (body.reference.mode === "manual" && !(body.reference.value <= 0 || body.reference.value > 0)) {
    setStatusError("手动 E_ref 不是有效数字");
    return;
  }
  if (LAST_PREVIEW && LAST_PREVIEW.dim_full > 1e6 &&
      !window.confirm(`该体系全空间维度 ${fmtInt(LAST_PREVIEW.dim_full)} (>1e6), 计算可能很慢/占内存。继续?`)) {
    return;
  }
  try {
    await api("/api/run", "POST", body);
    $("btn_run").disabled = true;
    $("btn_cancel").disabled = false;
    $("run_note").textContent = "";
    startPolling();
  } catch (e) {
    setStatusError(e.message);
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollOnce, 1200);
  pollOnce();
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}
async function pollOnce() {
  let j;
  try {
    j = await api("/api/job");
  } catch (e) {
    setStatusError("轮询失败: " + e.message);
    return;
  }
  if (j.status === "idle") { stopPolling(); resetButtons(); return; }
  if (j.status === "running") {
    renderProgress(j);
  } else {
    stopPolling();
    resetButtons();
    renderFinal(j);
  }
}
function resetButtons() {
  $("btn_run").disabled = false;
  $("btn_cancel").disabled = true;
}

function setStatusError(msg) {
  $("status_body").innerHTML = `<div class="err-box">${escapeHtml(msg)}</div>`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderProgress(j) {
  const ph = PHASE_TEXT[j.phase] || j.phase;
  let html = `<p class="status-line">阶段: <b>${ph}</b>` +
    (j.phase === "run" ? ` · seed ${j.seed_index + 1}/${j.n_seeds}` : "") + `</p>`;
  const frac = j.phase === "run"
    ? ((j.seed_index + (j.live ? 0.5 : 1)) / j.n_seeds) * 100
    : 5;
  html += `<div class="progress-track"><div class="progress-fill" style="width:${Math.min(100, frac)}%"></div></div>`;
  if (j.system_info) {
    const si = j.system_info;
    html += `<p class="hint">${escapeHtml(si.formula)} · (${si.na}α,${si.nb}β)@${si.norb}o · ` +
      `SCF ${si.scf_type} E=${si.e_scf?.toFixed(8)} (收敛: ${si.scf_converged ? "是" : "否"})</p>`;
  }
  if (j.reference && j.reference.mode === "auto" && !j.reference.skipped) {
    html += `<p class="hint">参考 (全空间 dim=${fmtInt(j.reference.dim)}) E_ref = ${fmtNum(j.reference.E_total, 12)}</p>`;
  }
  if (j.phase === "run" && j.live) {
    const lt = j.live;
    html += `<p class="status-line">轮次 <b>${lt.rounds_done}</b> · E = <b>${fmtNum(lt.last_E, 10)}</b>` +
      ` · dim = <b>${fmtInt(lt.last_dim)}</b> · σ² = ${fmtNum(lt.last_sigma2)}</p>`;
    $("status_body").innerHTML = html;
    $("results_card").hidden = false;
    lineChart($("results_body"), [{ name: "当前 seed", points: lt.trajectory.map((t, i) => ({ x: i, y: t.E })) }],
              { title: "E 收敛 (实时)", xLabel: "轨迹点", yLabel: "E (Ha)" });
  } else {
    $("status_body").innerHTML = html;
  }
}

function renderFinal(j) {
  if (j.status === "error") {
    setStatusError(j.error || "未知错误");
    return;
  }
  const cancelled = j.status === "cancelled";
  renderResults(j, cancelled);
}

function chip(k, v, s) {
  return `<div class="chip"><div class="k">${k}</div><div class="v">${v}</div>` +
    (s ? `<div class="s">${s}</div>` : "") + `</div>`;
}

function renderResults(j, partial) {
  const agg = j.aggregate;
  const results = j.seed_results || [];
  const ref = j.reference;
  const sysd = j.system_info || {};
  let html = "";

  if (partial) html += `<p class="hint">⚠ 任务被取消, 以下为已完成部分。</p>`;
  if (ref && ref.skipped) html += `<p class="hint">⚠ 参考计算跳过: ${escapeHtml(ref.note || "")}</p>`;

  if (agg && results.length) {
    html += `<div class="chips">`;
    html += chip("基态能量 E (Ha)",
      agg.n_seeds > 1 ? fmtNum(agg.E_mean, 10) : fmtNum(results[0].E_total, 10),
      agg.n_seeds > 1 ? `± ${fmtNum(agg.E_std, 2)} (n=${agg.n_seeds})` : `seed=${results[0].seed}`);
    if (agg.err_mean !== undefined) {
      html += chip("|E − E_ref|", fmtNum(agg.err_mean, 3),
        agg.n_seeds > 1 ? `max ${fmtNum(agg.err_max, 3)} · 最优 seed ${agg.best_seed}` : "");
    }
    html += chip("总 wall", `${agg.wall_total.toFixed(1)} s`,
      agg.n_seeds > 1 ? `${results.length} 个 seed 合计` : "");
    const last = results[results.length - 1];
    if (last.extras && last.extras.final_dim) html += chip("末态维度", fmtInt(last.extras.final_dim));
    if (last.extras && last.extras.e_pt2_final !== undefined && last.extras.e_pt2_final !== null)
      html += chip("末点 e_PT2", fmtNum(last.extras.e_pt2_final, 3));
    if (last.extras && last.extras.dim && last.method !== "sqd_active")
      html += chip("变分维度", fmtInt(last.extras.dim));
    html += `</div>`;
  }

  if (ref && !ref.skipped) {
    html += `<p class="hint">参考能量: ${ref.mode === "manual" ? "手动输入" :
      `solve_sci 全空间 (dim=${fmtInt(ref.dim)}, ${ref.wall_s?.toFixed(1)}s)`} · ` +
      `E_ref = ${fmtNum(ref.E_total, 12)} Ha</p>`;
  }
  html += `<p class="hint">体系: ${escapeHtml(sysd.formula || "")} · (${sysd.na}α,${sysd.nb}β)@${sysd.norb}o · ` +
    `SCF ${sysd.scf_type || ""} E=${sysd.e_scf?.toFixed(8) ?? "—"} Ha (含核/冻结核 ecore)</p>`;

  /* 轨迹图 (仅 sqd_active 有 trajectory) */
  const trajSeeds = results.filter((r) => r.trajectory && r.trajectory.length);
  if (trajSeeds.length) {
    const seriesOf = (key) => trajSeeds.map((r, i) => ({
      name: `seed ${r.seed}`, color: PALETTE[i % PALETTE.length],
      points: r.trajectory.map((t, k) => ({ x: k, y: t[key] })),
    }));
    html += `<div class="charts">`;
    html += `<div id="chart_E"></div><div id="chart_err"></div>`;
    html += `<div id="chart_dim"></div><div id="chart_sigma"></div>`;
    html += `</div>`;
    $("results_body").innerHTML = html;

    lineChart($("chart_E"), seriesOf("E"), { title: "能量收敛 E (Ha)", xLabel: "轨迹点 (末点=最终)" });
    if (ref && !ref.skipped) {
      barChart($("chart_err"), results.map((r, i) => ({ label: `seed ${r.seed}`, value: r.err_vs_ref })),
               { title: "|E − E_ref| 每 seed", yLabel: "err (Ha)", logY: true });
    } else {
      $("chart_err").innerHTML = "<div class='chart'><div class='title'>err 每 seed</div><p class='hint'>无参考能量 (选 auto/manual 参考后显示)</p></div>";
    }
    lineChart($("chart_dim"), seriesOf("dim"), { title: "对角化维度增长", xLabel: "轨迹点", yLabel: "dim", logY: true });
    lineChart($("chart_sigma"), seriesOf("sigma2"), { title: "σ² (子空间外方差)", xLabel: "轨迹点", yLabel: "σ²", logY: true });
  } else {
    $("results_body").innerHTML = html;
  }

  /* seed 表 */
  if (results.length) {
    let tbl = `<table class="seeds"><tr><th>seed</th><th>E_total (Ha)</th><th>err vs ref</th>` +
      `<th>wall (s)</th><th>dim</th><th>轮数</th></tr>`;
    for (const r of results) {
      tbl += `<tr><td>${r.seed}</td><td>${fmtNum(r.E_total, 10)}</td>` +
        `<td>${r.err_vs_ref === null ? "—" : fmtNum(r.err_vs_ref, 3)}</td>` +
        `<td>${r.wall_s.toFixed(1)}</td>` +
        `<td>${fmtInt(r.extras?.final_dim ?? r.extras?.dim ?? "—")}</td>` +
        `<td>${r.extras?.n_rounds ?? "—"}</td></tr>`;
    }
    tbl += `</table>`;
    $("results_body").insertAdjacentHTML("beforeend", tbl);
  }

  /* 原始 JSON */
  const raw = `<details class="raw"><summary>原始结果 JSON</summary><pre>` +
    escapeHtml(JSON.stringify(j, null, 2)) + `</pre></details>`;
  $("results_body").insertAdjacentHTML("beforeend", raw);
  $("results_card").hidden = false;
}

/* ---------------- 初始化 ---------------- */
async function init() {
  try {
    const caps = await api("/api/capabilities");
    $("caps").innerHTML =
      `<span class="badge ${caps.gpu ? "gpu-on" : "gpu-off"}">GPU ${caps.gpu ? "可用" : "不可用"}</span>` +
      `<span class="badge">tc_sqd ${caps.versions.tc_sqd}</span>` +
      `<span class="badge">pyscf ${caps.versions.pyscf}</span>`;
    if (!caps.gpu) $("backend").options[1].disabled = true;
  } catch (e) {
    $("caps").textContent = "capabilities 获取失败: " + e.message;
  }
  try {
    const { presets } = await api("/api/presets");
    PRESETS = presets;
    const sel = $("sys_preset");
    for (const p of presets) {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = p.label;
      sel.appendChild(o);
    }
    sel.onchange = () => applyPreset(presets.find((p) => p.id === sel.value));
    applyPreset(presets[0]);
  } catch (e) {
    $("preset_desc").textContent = "预设加载失败: " + e.message;
  }

  for (const id of ["sys_geometry", "sys_basis", "sys_charge", "sys_spin",
                    "sys_ncore", "sys_nvirt", "sys_scf"]) {
    $(id).addEventListener("change", () => { $("sys_preset").value = ""; refreshPreview(); });
  }
  $("method").addEventListener("change", refreshVisibility);
  $("sample_mode").addEventListener("change", refreshVisibility);
  $("multiseed").addEventListener("change", refreshVisibility);
  $("ref_mode").addEventListener("change", refreshVisibility);
  $("hf_weight").addEventListener("input", () => {
    $("hf_weight_val").textContent = (+$("hf_weight").value).toFixed(2);
  });
  $("btn_run").addEventListener("click", submitJob);
  $("btn_cancel").addEventListener("click", async () => {
    try { await api("/api/job/cancel", "POST", {}); } catch (e) { /* 忽略 */ }
    $("run_note").textContent = "已请求取消 (在当前 seed 结束后生效)";
  });
  refreshVisibility();
  pollOnce();   // 万一页面刷新时后端还有任务在跑, 直接接上
}

init();
