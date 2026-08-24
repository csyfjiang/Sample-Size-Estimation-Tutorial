/* Sample Size Estimation web demo -- runs sampsizeval in the browser via Pyodide. */

const PY_FILES = [
  "py/sampsizeval/__init__.py",
  "py/sampsizeval/development.py",
  "py/sampsizeval/validation_closed.py",
  "py/sampsizeval/validation_sim.py",
  "py/sampsizeval/compare_auc.py",
  "py/sampsizeval/data.py",
  "py/webapi.py",
];

let pyodide = null;
let ready = false;

const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");

function setStatus(text, ready) {
  statusText.textContent = text;
  statusEl.classList.toggle("ready", !!ready);
}

async function boot() {
  try {
    setStatus("Loading Python engine…");
    pyodide = await loadPyodide();
    setStatus("Loading numpy + scipy…");
    await pyodide.loadPackage(["numpy", "scipy"]);

    setStatus("Loading calculators…");
    pyodide.FS.mkdirTree("/home/pyodide/sampsizeval");
    for (const path of PY_FILES) {
      const src = await (await fetch(path + "?v=1")).text();
      const dest = "/home/pyodide/" + path.replace(/^py\//, "");
      pyodide.FS.writeFile(dest, src);
    }
    await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/home/pyodide")
from webapi import (development, validation_closed, validation_sim, compare,
                    params_from_predictions, params_from_two_models)
`);

    ready = true;
    setStatus("Ready", true);
    document.querySelectorAll("button.run").forEach(b => (b.disabled = false));
  } catch (e) {
    setStatus("Failed to load: " + e.message);
    console.error(e);
  }
}

/* ---- python call helper -------------------------------------------------- */
function pyCall(fn, kwargs) {
  const f = pyodide.globals.get(fn);
  const res = f.callKwargs(kwargs);
  const obj = res.toJs({ dict_converter: Object.fromEntries });
  res.destroy();
  f.destroy();
  return obj;
}

/* ---- formatting ---------------------------------------------------------- */
const fmt = (x, nd = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(nd);
const fmt0 = x => (x === null || x === undefined || Number.isNaN(x) ? "—" : Math.round(x).toLocaleString());

function badge(ok, warn) {
  if (ok) return '<span class="badge ok">OK</span>';
  if (warn) return '<span class="badge warn">borderline</span>';
  return '<span class="badge bad">wide</span>';
}
function rowsTable(head, rows) {
  return `<table class="res"><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r => `<tr>${r.map((c, i) => `<td class="${i > 0 && i < r.length - 1 ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

/* ---- renderers ----------------------------------------------------------- */
function renderDev(r) {
  const rows = [
    ["B1 &mdash; margin of error", fmt(r.B1_margin, 4), fmt0(r.B1_required_n), badge(r.B1_ok)],
    ["B2 &mdash; MAPE", fmt(r.B2_mape, 4), fmt0(r.B2_required_n), badge(r.B2_ok)],
    ["B3 &mdash; shrinkage S", fmt(r.B3_shrinkage), fmt0(r.B3_required_n), badge(r.B3_ok)],
    ["B4 &mdash; optimism", fmt(r.B4_optimism, 4), fmt0(r.B4_required_n), badge(r.B4_ok)],
  ];
  const suff = r.sufficient;
  const verdict = suff
    ? `<div class="verdict ok">Sufficient. Required N = ${fmt0(r.overall_required_n)}; you have ${fmt0(r.n)}.</div>`
    : `<div class="verdict bad">Short by ${fmt0(r.shortfall)}. Required N = ${fmt0(r.overall_required_n)} (binding: ${r.binding_criterion}); you have ${fmt0(r.n)}.</div>`;
  return `<h2 style="margin-top:0">Development &mdash; N = ${fmt0(r.n)}</h2>
    <p class="desc">max(R&sup2;<sub>cs</sub>) = ${fmt(r.max_r2cs)} &middot; implied R&sup2;<sub>Nagelkerke</sub> = ${fmt(r.r2_nagelkerke)}</p>
    ${rowsTable(["Criterion", "Achieved", "N needed", "Verdict"], rows)}${verdict}`;
}

function renderVC(r) {
  const nb = Object.entries(r.net_benefit || {}).map(([t, d]) =>
    [`Net benefit @ p<sub>t</sub>=${t}`, fmt(d.ci_width, 3), "&mdash;", ""]);
  const rows = [
    ["O/E ratio", fmt(r.oe_ci_width), "&le; 0.22", badge(r.oe_ok, r.oe_ci_width <= 0.35)],
    ["Calibration slope", fmt(r.calibration_slope_ci_width), "&le; 0.30", badge(r.calibration_slope_ok, r.calibration_slope_ci_width <= 0.5)],
    ["c-statistic", fmt(r.c_stat_ci_width), "&le; 0.10", badge(r.c_stat_ok, r.c_stat_ci_width <= 0.15)],
    ...nb,
  ];
  const bind = !r.calibration_slope_ok;
  const verdict = `<div class="verdict ${r.c_stat_ok && r.calibration_slope_ok ? "ok" : "bad"}">
    ${bind ? "Calibration slope is the binding constraint (wide CI)." : "All primary precision targets met."}</div>`;
  return `<h2 style="margin-top:0">Validation (closed form) &mdash; N = ${fmt0(r.n)}</h2>
    <p class="desc">95% CI widths at &phi;=${fmt(r.phi, 2)}, c=${fmt(r.c_stat, 2)}</p>
    ${rowsTable(["Measure", "95% CI width", "Target", "Verdict"], rows)}${verdict}`;
}

function renderVS(r) {
  const nb = Object.entries(r.net_benefit || {}).map(([t, d]) =>
    [`Net benefit @ p<sub>t</sub>=${t}`, fmt(d.mean, 4), fmt(d.ci_width, 4)]);
  const rows = [
    ["c-statistic", fmt(r.c_statistic_mean), fmt(r.c_statistic_ci_width)],
    ["Calibration slope", fmt(r.calibration_slope_mean), fmt(r.calibration_slope_ci_width)],
    ["Calibration-in-large", fmt(r.citl_mean), fmt(r.citl_ci_width)],
    ["O/E ratio", fmt(r.oe_mean), fmt(r.oe_ci_width)],
    ...nb,
  ];
  const cal = r.S === 1 && r.gamma === 0;
  return `<h2 style="margin-top:0">Validation (simulation) &mdash; N = ${fmt0(r.n)}</h2>
    <p class="desc">${cal ? "Well calibrated (&gamma;=0, S=1)" : `Miscalibrated (&gamma;=${fmt(r.gamma, 2)}, S=${fmt(r.S, 2)})`} &middot; mean events &asymp; ${fmt(r.mean_events, 1)}</p>
    ${rowsTable(["Measure", "Mean estimate", "Avg 95% CI width"], rows)}`;
}

function renderCmp(o) {
  const req = o.required;
  let html = `<h2 style="margin-top:0">Compare AUCs</h2>
    <p class="desc">&theta;&#8321;=${fmt(req.theta1, 2)}, &theta;&#8322;=${fmt(req.theta2, 2)}, &Delta;=${fmt(Math.abs(req.diff))}, &rho;=${fmt(req.rho, 2)}, &gamma;=${fmt(req.gamma, 2)}</p>`;
  html += rowsTable(["Quantity", "Value", ""], [
    ["Required total N (power " + fmt(req.power, 2) + ")", fmt0(req.N), ""],
    ["&nbsp;&nbsp;&mdash; cases / controls", `${fmt0(req.m)} / ${fmt0(req.n)}`, ""],
  ]);
  if (o.achieved) {
    const a = o.achieved;
    const cls = a.power >= 0.8 ? "ok" : "bad";
    html += `<div class="verdict ${cls}">At your N = ${fmt0(a.N)}: achieved power = ${fmt(a.power)} &middot; ${a.verdict}</div>`;
    if (o.mdd && o.mdd.theta2_min) {
      html += `<p class="desc" style="margin-top:12px">Minimum detectable &theta;&#8322; at this N (power ${fmt(o.mdd.power, 2)}): <b>${fmt(o.mdd.theta2_min)}</b> (min detectable &Delta; = ${fmt(o.mdd.min_detectable_diff)}).</p>`;
    }
  }
  return html;
}

/* ---- run buttons --------------------------------------------------------- */
function num(id) { return parseFloat(document.getElementById(id).value); }
function setBusy(outId, msg) {
  document.getElementById(outId).innerHTML = `<div class="results-empty"><span class="spin"></span> ${msg}</div>`;
}

const RUNNERS = {
  dev(out) {
    setBusy(out, "Calculating…");
    const r2raw = document.getElementById("dev-r2").value;
    const kw = { n: num("dev-n"), phi: num("dev-phi"), P: num("dev-P") };
    if (r2raw !== "") kw.r2cs = parseFloat(r2raw);
    const r = pyCall("development", kw);
    document.getElementById(out).innerHTML = renderDev(r);
  },
  vc(out) {
    setBusy(out, "Calculating (includes a short simulation for the slope)…");
    setTimeout(() => {
      const r = pyCall("validation_closed", { n: num("vc-n"), phi: num("vc-phi"), c_stat: num("vc-c") });
      document.getElementById(out).innerHTML = renderVC(r);
    }, 20);
  },
  vs(out) {
    setBusy(out, "Running simulation…");
    setTimeout(() => {
      const r = pyCall("validation_sim", {
        n: num("vs-n"), mu: num("vs-mu"), sigma: num("vs-sigma"),
        gamma: num("vs-gamma"), S: num("vs-S"), n_sims: num("vs-nsims"),
      });
      document.getElementById(out).innerHTML = renderVS(r);
    }, 20);
  },
  cmp(out) {
    setBusy(out, "Integrating…");
    setTimeout(() => {
      const o = pyCall("compare", {
        n: num("cmp-n"), theta1: num("cmp-t1"), theta2: num("cmp-t2"),
        rho: num("cmp-rho"), gamma: num("cmp-gamma"),
        alpha: num("cmp-alpha"), power: num("cmp-power"),
      });
      document.getElementById(out).innerHTML = renderCmp(o);
    }, 20);
  },
};

document.querySelectorAll("button.run").forEach(btn => {
  btn.addEventListener("click", () => {
    const which = btn.dataset.run;
    try { RUNNERS[which]("out-" + which); }
    catch (e) { document.getElementById("out-" + which).innerHTML = `<div class="verdict bad">Error: ${e.message}</div>`; console.error(e); }
  });
});

/* ---- tabs ---------------------------------------------------------------- */
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
  });
});

/* ---- CSV parsing + dropzones -------------------------------------------- */
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map(h => h.trim());
  const cols = {};
  header.forEach(h => (cols[h] = []));
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const parts = lines[i].split(",");
    header.forEach((h, j) => cols[h].push(parts[j]));
  }
  return cols;
}
const toNums = a => a.map(Number);

const DROP_HANDLERS = {
  "dev-outcome"(cols, msgEl) {
    const y = toNums(cols.true_outcome);
    const phi = y.reduce((s, v) => s + v, 0) / y.length;
    document.getElementById("dev-phi").value = phi.toFixed(3);
    msgEl.innerHTML = `Loaded N=${y.length}. Set &phi;=${phi.toFixed(3)}.`;
  },
  "vc-pred"(cols, msgEl) {
    const p = pyCall("params_from_predictions", { prob: toNums(cols.predicted_prob), outcome: toNums(cols.true_outcome) });
    document.getElementById("vc-phi").value = p.phi.toFixed(3);
    document.getElementById("vc-c").value = p.c_stat.toFixed(3);
    msgEl.innerHTML = `Loaded N=${p.n}, events=${p.events}. &phi;=${p.phi.toFixed(3)}, c=${p.c_stat.toFixed(3)}.`;
  },
  "vs-pred"(cols, msgEl) {
    const p = pyCall("params_from_predictions", { prob: toNums(cols.predicted_prob), outcome: toNums(cols.true_outcome) });
    document.getElementById("vs-mu").value = p.mu.toFixed(3);
    document.getElementById("vs-sigma").value = p.sigma.toFixed(3);
    msgEl.innerHTML = `Loaded N=${p.n}. &mu;=${p.mu.toFixed(3)}, &sigma;=${p.sigma.toFixed(3)}.`;
  },
  "cmp-two"(cols, msgEl) {
    const p = pyCall("params_from_two_models", {
      score_a: toNums(cols.model_A_score), score_b: toNums(cols.model_B_score), outcome: toNums(cols.true_outcome),
    });
    document.getElementById("cmp-t1").value = p.theta1.toFixed(3);
    document.getElementById("cmp-t2").value = p.theta2.toFixed(3);
    document.getElementById("cmp-rho").value = p.rho.toFixed(3);
    document.getElementById("cmp-gamma").value = p.gamma.toFixed(3);
    document.getElementById("cmp-n").value = p.n;
    msgEl.innerHTML = `Loaded N=${p.n}. &theta;&#8321;=${p.theta1.toFixed(3)}, &theta;&#8322;=${p.theta2.toFixed(3)}, &rho;=${p.rho.toFixed(3)}, &gamma;=${p.gamma.toFixed(3)}.`;
  },
};

document.querySelectorAll(".drop").forEach(zone => {
  const key = zone.dataset.drop;
  const msgEl = zone.querySelector(".filemsg");
  const handle = file => {
    const reader = new FileReader();
    reader.onload = () => {
      if (!ready) { msgEl.textContent = "Engine still loading, try again in a moment."; return; }
      try { DROP_HANDLERS[key](parseCSV(reader.result), msgEl); }
      catch (e) { msgEl.textContent = "Could not parse: " + e.message; console.error(e); }
    };
    reader.readAsText(file);
  };
  zone.addEventListener("click", () => {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = ".csv,text/csv";
    inp.onchange = () => inp.files[0] && handle(inp.files[0]);
    inp.click();
  });
  ["dragover", "dragenter"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add("hot"); }));
  ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove("hot"); }));
  zone.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) handle(f); });
});

boot();
