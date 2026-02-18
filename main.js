/* ═══════════════════════════════════════════════
   BADMOSH CODERS — Duality AI Segmentation
   main.js
═══════════════════════════════════════════════ */

"use strict";

/* ── DATA ─────────────────────────────────────── */

const CLASS_DATA = [
  { name: "Background",     iou: 1.0000, color: "#555568", weight: 0.5  },
  { name: "Trees",          iou: 0.7815, color: "#2d6b3a", weight: 1.0  },
  { name: "Lush Bushes",    iou: 0.6599, color: "#39e075", weight: 1.0  },
  { name: "Dry Grass",      iou: 0.6669, color: "#f5c842", weight: 1.0  },
  { name: "Dry Bushes",     iou: 0.4697, color: "#8b6914", weight: 1.5  },
  { name: "Ground Clutter", iou: 0.3222, color: "#6b6050", weight: 2.0  },
  { name: "Flowers",        iou: 0.5391, color: "#ff4d1c", weight: 3.0  },
  { name: "Logs",           iou: 0.3198, color: "#7a4a2a", weight: 3.0  },
  { name: "Rocks",          iou: 0.4653, color: "#888080", weight: 1.5  },
  { name: "Landscape",      iou: 0.4699, color: "#d4852a", weight: 0.5  },
  { name: "Sky",            iou: 0.9804, color: "#4da6ff", weight: 0.5  },
];

const TRAINING_DATA = [
  { epoch: 1,  loss: 2.41, iou: 0.21 },
  { epoch: 2,  loss: 1.98, iou: 0.28 },
  { epoch: 3,  loss: 1.62, iou: 0.33 },
  { epoch: 4,  loss: 1.38, iou: 0.37 },
  { epoch: 5,  loss: 1.19, iou: 0.40 },
  { epoch: 6,  loss: 1.04, iou: 0.43 },
  { epoch: 7,  loss: 0.93, iou: 0.46 },
  { epoch: 8,  loss: 0.85, iou: 0.48 },
  { epoch: 9,  loss: 0.79, iou: 0.49 },
  { epoch: 10, loss: 0.74, iou: 0.50 },
];

const IMPROVEMENTS = [
  {
    num: "01", title: "Larger Backbone", tag: "+EMBEDDING DIM",
    desc: "Switched from ViT-S/14 (384-dim) to ViT-B/14 (768-dim) for richer feature extraction from desert terrain textures and structures."
  },
  {
    num: "02", title: "Partial Unfreeze", tag: "+DOMAIN ADAPT",
    desc: "Last 3 transformer blocks fine-tuned at lr=1e-5. The backbone adapts to desert domain while preserving millions of images of pretrained knowledge."
  },
  {
    num: "03", title: "Class Weighting", tag: "+RARE CLASS",
    desc: "Flowers and Logs get 3× weight. Sky and Landscape get 0.5×. Stops the model from ignoring rare classes that cover few pixels."
  },
  {
    num: "04", title: "Augmentation", tag: "+GENERALIZATION",
    desc: "Synchronized horizontal and vertical flips on both image and mask. Color jitter on image only. Reduces overfitting to the training biome."
  },
  {
    num: "05", title: "Deeper Head", tag: "+CAPACITY",
    desc: "Segmentation head expanded from 3 to 5 convolutional layers (512→256→256→128→11), increasing its capacity to decode complex terrain features."
  },
  {
    num: "06", title: "Full Resolution", tag: "+RESOLUTION",
    desc: "Input resolution doubled from 476×266 to 952×532 for finer boundary details and better small-object segmentation for rare classes."
  },
];

/* ── CURSOR ────────────────────────────────────── */

(function initCursorDISABLED() {
  const cursor = document.getElementById("cursor");
  const trail  = document.getElementById("cursor-trail");
  let tx = 0, ty = 0, cx = 0, cy = 0;

  document.addEventListener("mousemove", (e) => {
    tx = e.clientX; ty = e.clientY;
    cursor.style.left = tx + "px";
    cursor.style.top  = ty + "px";
  });

  function animateTrail() {
    cx += (tx - cx) * 0.12;
    cy += (ty - cy) * 0.12;
    trail.style.left = cx + "px";
    trail.style.top  = cy + "px";
    requestAnimationFrame(animateTrail);
  }
  animateTrail();

  document.querySelectorAll("a, button, .banner-item, .class-card, .imp-card, .pipe-node").forEach((el) => {
    el.addEventListener("mouseenter", () => {
      cursor.style.width  = "14px";
      cursor.style.height = "14px";
      trail.style.width   = "42px";
      trail.style.height  = "42px";
    });
    el.addEventListener("mouseleave", () => {
      cursor.style.width  = "";
      cursor.style.height = "";
      trail.style.width   = "";
      trail.style.height  = "";
    });
  });
})();

/* ── SCROLL REVEAL ─────────────────────────────── */

(function initReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll("[data-reveal]").forEach((el) => obs.observe(el));
})();

/* ── ACTIVE NAV ─────────────────────────────────── */

(function initNav() {
  const sections = document.querySelectorAll("section[id]");
  const links    = document.querySelectorAll(".nav-link");

  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        links.forEach((l) => l.classList.remove("active"));
        const active = document.querySelector(`.nav-link[data-section="${e.target.id}"]`);
        if (active) active.classList.add("active");
      }
    });
  }, { rootMargin: "-40% 0px -55%", threshold: 0 });

  sections.forEach((s) => obs.observe(s));
})();

/* ── COUNT-UP ANIMATION ─────────────────────────── */

function countUp(el, target, duration = 1200, delay = 0) {
  setTimeout(() => {
    const start = performance.now();
    const isFloat = target < 10 && String(target).includes(".");

    function tick(now) {
      const p    = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      const val  = target * ease;
      el.textContent = isFloat ? val.toFixed(4) : Math.round(val).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = isFloat ? target.toFixed(4) : target.toLocaleString();
    }
    requestAnimationFrame(tick);
  }, delay);
}

// Hero stats count-up
(function initHeroStats() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.querySelectorAll("[data-count]").forEach((el, i) => {
          countUp(el, parseInt(el.dataset.count), 1400, i * 100);
        });
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });

  const heroStats = document.querySelector(".hero-stats");
  if (heroStats) obs.observe(heroStats);
})();

// Banner mIoU count-up
(function initBannerMiou() {
  const el  = document.getElementById("banner-miou");
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        countUp(el, 0.5668, 1600, 200);
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.5 });
  if (el) obs.observe(el.closest(".banner-item"));
})();

/* ── TERRAIN CANVAS ─────────────────────────────── */

(function drawTerrain() {
  const canvas = document.getElementById("terrain-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  // Color palette matching class colors
  const PALETTE = [
    "#555568", "#2d6b3a", "#39e075", "#f5c842",
    "#8b6914", "#6b6050", "#ff4d1c", "#7a4a2a",
    "#888080", "#d4852a", "#4da6ff"
  ];

  // Generate pixel grid segmentation art
  const CELL = 14;
  const cols = Math.ceil(W / CELL);
  const rows = Math.ceil(H / CELL);

  // Simplex-like noise using sin/cos layering
  function noise(x, y, seed) {
    return (
      Math.sin(x * 0.15 + seed) * 0.5 +
      Math.sin(y * 0.12 + seed * 1.3) * 0.3 +
      Math.sin((x + y) * 0.08 + seed * 0.7) * 0.2
    );
  }

  // Assign each cell a class based on position + noise
  function getClass(cx, cy) {
    const n = noise(cx, cy, 3.7);
    const yRatio = cy / rows;

    if (yRatio < 0.12) return 10; // Sky
    if (yRatio < 0.18 && n > 0.1) return 0; // Background / horizon
    if (yRatio > 0.82) return 9; // Landscape ground
    if (yRatio > 0.7 && n > 0.3)  return 8; // Rocks
    if (yRatio > 0.65 && n < -0.2) return 7; // Logs
    if (n > 0.55) return 1; // Trees
    if (n > 0.35) return 2; // Lush Bushes
    if (n > 0.15) return 3; // Dry Grass
    if (n > -0.05) return 4; // Dry Bushes
    if (n > -0.25) return 5; // Ground Clutter
    if (n > -0.45) return 6; // Flowers
    return 9; // Landscape
  }

  // Draw segmentation grid
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cls   = getClass(c, r);
      const color = PALETTE[cls];
      ctx.fillStyle = color;
      ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1);
    }
  }

  // Overlay grid lines
  ctx.strokeStyle = "rgba(6,6,8,0.5)";
  ctx.lineWidth = 1;
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath();
    ctx.moveTo(c * CELL, 0);
    ctx.lineTo(c * CELL, H);
    ctx.stroke();
  }
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * CELL);
    ctx.lineTo(W, r * CELL);
    ctx.stroke();
  }

  // Scan animation — moving highlight line
  let scanY = 0;
  function animate() {
    // Draw scan line
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = "#ff4d1c";
    ctx.fillRect(0, scanY - 2, W, 4);
    ctx.restore();

    scanY += 1.5;
    if (scanY > H) scanY = 0;
    requestAnimationFrame(animate);
  }
  animate();
})();

/* ── CLASS CARDS ─────────────────────────────────── */

(function buildClassGrid() {
  const grid = document.getElementById("class-grid");
  if (!grid) return;

  CLASS_DATA.forEach((cls) => {
    const pct  = Math.round(cls.iou * 100);
    const tag  = cls.iou < 0.40 ? "RARE — LOW IoU" : cls.iou > 0.70 ? "GOOD" : "OK";
    const tagClass = cls.iou < 0.40 ? "tag-rare" : cls.iou > 0.70 ? "tag-good" : "tag-ok";

    const card = document.createElement("div");
    card.className = "class-card";
    card.style.setProperty("--color", cls.color);

    card.innerHTML = `
      <div class="cc-name">${cls.name.toUpperCase()}</div>
      <div class="cc-bar-track">
        <div class="cc-bar-fill" style="--color:${cls.color}"></div>
      </div>
      <div class="cc-row">
        <span class="cc-pct">${pct}<span style="font-size:14px;color:var(--dim)">%</span></span>
        <span class="cc-weight">weight ×${cls.weight}</span>
      </div>
      <span class="cc-tag ${tagClass}">${tag}</span>
    `;
    grid.appendChild(card);
  });

  // Animate bars on scroll
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.querySelectorAll(".cc-bar-fill").forEach((bar, i) => {
          const pct = Math.round(CLASS_DATA[i].iou * 100);
          setTimeout(() => { bar.style.width = pct + "%"; }, i * 60);
        });
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.2 });

  obs.observe(grid);
})();

/* ── IMPROVEMENT CARDS ───────────────────────────── */

(function buildImpGrid() {
  const grid = document.getElementById("imp-grid");
  if (!grid) return;

  IMPROVEMENTS.forEach((imp) => {
    const card = document.createElement("div");
    card.className = "imp-card";
    card.setAttribute("data-reveal", "");

    card.innerHTML = `
      <div class="imp-title">${imp.title}</div>
      <div class="imp-desc">${imp.desc}</div>
      <div class="imp-tag">${imp.tag}</div>
      <div class="imp-num">${imp.num}</div>
    `;
    grid.appendChild(card);
  });

  // Re-observe newly added elements
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  grid.querySelectorAll("[data-reveal]").forEach((el) => obs.observe(el));
})();

/* ── CHARTS ──────────────────────────────────────── */

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "#111116",
      borderColor: "#1e1e26",
      borderWidth: 1,
      titleColor: "#eeeef8",
      bodyColor: "#555568",
      titleFont: { family: "'DM Mono', monospace", size: 10 },
      bodyFont:  { family: "'DM Mono', monospace", size: 10 },
      padding: 12,
      callbacks: {
        title: (items) => `EPOCH ${items[0].label}`,
      },
    },
  },
  scales: {
    x: {
      grid:  { color: "rgba(30,30,38,0.8)", drawBorder: false },
      ticks: { color: "#555568", font: { family: "'DM Mono', monospace", size: 9 } },
    },
    y: {
      grid:  { color: "rgba(30,30,38,0.8)", drawBorder: false },
      ticks: { color: "#555568", font: { family: "'DM Mono', monospace", size: 9 } },
    },
  },
};

// Loss Chart
new Chart(document.getElementById("loss-chart"), {
  type: "line",
  data: {
    labels: TRAINING_DATA.map((d) => d.epoch),
    datasets: [{
      data: TRAINING_DATA.map((d) => d.loss),
      borderColor: "#ff4d1c",
      backgroundColor: "rgba(255,77,28,0.07)",
      borderWidth: 2,
      pointRadius: 4,
      pointBackgroundColor: "#ff4d1c",
      pointBorderColor: "#060608",
      pointBorderWidth: 2,
      tension: 0.4,
      fill: true,
    }],
  },
  options: {
    ...CHART_DEFAULTS,
    scales: {
      ...CHART_DEFAULTS.scales,
      y: {
        ...CHART_DEFAULTS.scales.y,
        title: {
          display: true, text: "LOSS",
          color: "#555568",
          font: { family: "'DM Mono', monospace", size: 9 },
        },
      },
    },
  },
});

// IoU Chart
new Chart(document.getElementById("iou-chart"), {
  type: "line",
  data: {
    labels: TRAINING_DATA.map((d) => d.epoch),
    datasets: [{
      data: TRAINING_DATA.map((d) => d.iou),
      borderColor: "#39e075",
      backgroundColor: "rgba(57,224,117,0.07)",
      borderWidth: 2,
      pointRadius: 4,
      pointBackgroundColor: "#39e075",
      pointBorderColor: "#060608",
      pointBorderWidth: 2,
      tension: 0.4,
      fill: true,
    }],
  },
  options: {
    ...CHART_DEFAULTS,
    scales: {
      ...CHART_DEFAULTS.scales,
      y: {
        ...CHART_DEFAULTS.scales.y,
        min: 0, max: 1,
        title: {
          display: true, text: "mIoU",
          color: "#555568",
          font: { family: "'DM Mono', monospace", size: 9 },
        },
      },
    },
  },
});

/* ── HEADER SCROLL EFFECT ────────────────────────── */

(function initHeaderScroll() {
  const hdr = document.getElementById("hdr");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      hdr.style.borderBottomColor = "rgba(255,77,28,0.2)";
    } else {
      hdr.style.borderBottomColor = "";
    }
  }, { passive: true });
})();

console.log(
  "%c BADMOSH CODERS %c Duality AI Hackathon 2025 ",
  "background:#ff4d1c;color:#fff;font-family:monospace;font-size:13px;padding:4px 8px;font-weight:bold",
  "background:#111116;color:#555568;font-family:monospace;font-size:13px;padding:4px 8px"
);
