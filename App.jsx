import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from "recharts";

/* ═══════════════════════════════════════════════
   DATA
═══════════════════════════════════════════════ */

const CLASS_DATA = [
  { name: "Background",     iou: 0.62, color: "#555568", weight: 0.5  },
  { name: "Trees",          iou: 0.71, color: "#2d6b3a", weight: 1.0  },
  { name: "Lush Bushes",    iou: 0.58, color: "#39e075", weight: 1.0  },
  { name: "Dry Grass",      iou: 0.65, color: "#f5c842", weight: 1.0  },
  { name: "Dry Bushes",     iou: 0.54, color: "#8b6914", weight: 1.5  },
  { name: "Ground Clutter", iou: 0.38, color: "#6b6050", weight: 2.0  },
  { name: "Flowers",        iou: 0.29, color: "#ff4d1c", weight: 3.0  },
  { name: "Logs",           iou: 0.33, color: "#7a4a2a", weight: 3.0  },
  { name: "Rocks",          iou: 0.61, color: "#888080", weight: 1.5  },
  { name: "Landscape",      iou: 0.74, color: "#d4852a", weight: 0.5  },
  { name: "Sky",            iou: 0.82, color: "#4da6ff", weight: 0.5  },
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
  { num:"01", title:"Larger Backbone",  tag:"+EMBEDDING DIM", desc:"Switched from ViT-S/14 (384-dim) to ViT-B/14 (768-dim) for richer feature extraction from desert terrain." },
  { num:"02", title:"Partial Unfreeze", tag:"+DOMAIN ADAPT",  desc:"Last 3 transformer blocks fine-tuned at lr=1e-5, adapting to desert domain while preserving pretrained knowledge." },
  { num:"03", title:"Class Weighting",  tag:"+RARE CLASS",    desc:"Flowers and Logs get 3× weight. Sky and Landscape get 0.5×. Stops the model ignoring rare classes." },
  { num:"04", title:"Augmentation",     tag:"+GENERALIZATION",desc:"Synchronized flips on image and mask. Color jitter on image only. Reduces overfitting to training biome." },
  { num:"05", title:"Deeper Head",      tag:"+CAPACITY",      desc:"Segmentation head expanded from 3 to 5 convolutional layers (512→256→256→128→11)." },
  { num:"06", title:"Full Resolution",  tag:"+RESOLUTION",    desc:"Input resolution doubled from 476×266 to 952×532 for finer boundary details and better small-object segmentation." },
];

/* ═══════════════════════════════════════════════
   HOOKS
═══════════════════════════════════════════════ */

function useCountUp(target, duration = 1400, delay = 400) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => {
      const start = performance.now();
      const tick = (now) => {
        const p    = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        setValue(+(target * ease).toFixed(4));
        if (p < 1) requestAnimationFrame(tick);
        else setValue(target);
      };
      requestAnimationFrame(tick);
    }, delay);
    return () => clearTimeout(t);
  }, [target, duration, delay]);
  return value;
}

function useInView(ref, threshold = 0.15) {
  const [inView, setInView] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setInView(true); obs.disconnect(); } },
      { threshold }
    );
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, [threshold]);
  return inView;
}

function useStaggeredReveal(count, delay = 80) {
  const [visible, setVisible] = useState([]);
  const trigger = useCallback(() => {
    for (let i = 0; i < count; i++) {
      setTimeout(() => setVisible((v) => [...v, i]), i * delay);
    }
  }, [count, delay]);
  return [visible, trigger];
}

/* ═══════════════════════════════════════════════
   STYLES
═══════════════════════════════════════════════ */

const S = {
  app: {
    background: "#060608", color: "#c4c4d4",
    fontFamily: "'Barlow Condensed', sans-serif",
    minHeight: "100vh", overflowX: "hidden",
  },
  // Header
  hdr: {
    position: "sticky", top: 0, zIndex: 100,
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0 36px", height: 56,
    background: "rgba(6,6,8,0.95)", backdropFilter: "blur(16px)",
    borderBottom: "1px solid #1e1e26",
    fontFamily: "'DM Mono', monospace",
  },
  hdrLeft: { display: "flex", alignItems: "center", gap: 18 },
  badge: {
    display: "flex", alignItems: "center", gap: 6,
    padding: "4px 10px", border: "1px solid #ff4d1c",
    fontSize: 9, color: "#ff4d1c", letterSpacing: ".18em",
  },
  badgeDot: {
    width: 6, height: 6, borderRadius: "50%",
    background: "#ff4d1c", boxShadow: "0 0 8px #ff4d1c",
  },
  brandTeam: { color: "#eeeef8", fontWeight: 500, letterSpacing: ".15em", fontSize: 10 },
  brandSep:  { color: "#ff4d1c", fontSize: 10 },
  brandEvent:{ color: "#555568", letterSpacing: ".1em", fontSize: 10 },
  miouWrap:  { display: "flex", flexDirection: "column", alignItems: "flex-end" },
  miouLabel: { fontSize: 8, color: "#555568", letterSpacing: ".18em" },
  miouVal:   { fontSize: 18, color: "#ff4d1c", fontWeight: 500, lineHeight: 1.1 },

  // Hero
  hero: {
    minHeight: "100vh", display: "grid",
    gridTemplateColumns: "1fr 520px", alignItems: "center",
    padding: "80px 36px 60px", gap: 40, position: "relative",
  },
  heroTag: {
    display: "inline-flex", alignItems: "center", gap: 8,
    padding: "6px 14px", border: "1px solid #2a2a36",
    fontFamily: "'DM Mono', monospace", fontSize: 10,
    color: "#555568", letterSpacing: ".18em", marginBottom: 28,
  },
  heroTitle: {
    fontFamily: "'Bebas Neue', sans-serif",
    fontSize: "clamp(72px, 10vw, 120px)",
    lineHeight: .88, letterSpacing: "-.01em", marginBottom: 24,
  },
  heroSub: { fontSize: 15, color: "#555568", lineHeight: 1.7, maxWidth: 440, marginBottom: 36 },
  heroStats: { display: "flex", alignItems: "center" },
  hstatNum: {
    display: "block",
    fontFamily: "'Bebas Neue', sans-serif",
    fontSize: 42, color: "#eeeef8", lineHeight: 1,
  },
  hstatLabel: {
    fontFamily: "'DM Mono', monospace", fontSize: 9,
    color: "#555568", letterSpacing: ".18em", textTransform: "uppercase",
  },

  // Banner
  banner: {
    display: "flex", alignItems: "stretch",
    borderTop: "1px solid #1e1e26", borderBottom: "1px solid #1e1e26",
    background: "#111116",
  },
  bannerItem: { flex: 1, padding: "30px 26px", transition: "background .2s", cursor: "default" },
  biLabel: { fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568", letterSpacing: ".2em", textTransform: "uppercase", marginBottom: 10 },
  biVal:   { fontFamily: "'Bebas Neue', sans-serif", fontSize: 52, lineHeight: 1, color: "#eeeef8" },
  biUnit:  { fontSize: 22, color: "#555568", marginLeft: 2 },
  biSub:   { fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568", letterSpacing: ".12em", marginTop: 8 },

  // Section
  section:    { padding: "80px 36px" },
  secEyebrow: { fontFamily: "'DM Mono', monospace", fontSize: 10, color: "#ff4d1c", letterSpacing: ".22em", textTransform: "uppercase", marginBottom: 10 },
  secTitle:   { fontFamily: "'Bebas Neue', sans-serif", fontSize: "clamp(40px,5vw,64px)", color: "#eeeef8", lineHeight: 1, marginBottom: 48 },

  // Class card
  classGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: 1, background: "#1e1e26" },
  classCard: { background: "#111116", padding: 22, position: "relative", overflow: "hidden", cursor: "default" },
  ccName:  { fontFamily: "'DM Mono', monospace", fontSize: 10, color: "#555568", letterSpacing: ".15em", textTransform: "uppercase", marginBottom: 12, paddingLeft: 10 },
  ccTrack: { height: 4, background: "#1e1e26", marginBottom: 8, overflow: "hidden" },
  ccPct:   { fontFamily: "'Bebas Neue', sans-serif", fontSize: 28, color: "#eeeef8", lineHeight: 1 },
  ccRow:   { display: "flex", justifyContent: "space-between", alignItems: "baseline", paddingLeft: 10 },
  ccWeight:{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568" },

  // Charts
  chartsGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, background: "#1e1e26" },
  chartCard:  { background: "#111116", padding: 28 },
  ccTitle:    { fontFamily: "'DM Mono', monospace", fontSize: 10, color: "#ff4d1c", letterSpacing: ".18em", marginBottom: 18 },

  // Improvements
  impGrid: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 1, background: "#1e1e26" },
  impCard: { background: "#111116", padding: 28, position: "relative", overflow: "hidden", cursor: "default" },
  impTitle:{ fontFamily: "'DM Mono', monospace", fontSize: 11, fontWeight: 500, color: "#eeeef8", letterSpacing: ".08em", textTransform: "uppercase", marginBottom: 8 },
  impDesc: { fontSize: 13, color: "#555568", lineHeight: 1.65, maxWidth: 280 },
  impTag:  { display: "inline-block", marginTop: 12, padding: "3px 9px", border: "1px solid #ff4d1c", color: "#ff4d1c", fontFamily: "'DM Mono', monospace", fontSize: 8, letterSpacing: ".12em" },
  impNum:  { position: "absolute", right: 16, bottom: 8, fontFamily: "'Bebas Neue', sans-serif", fontSize: 64, color: "#1e1e26", lineHeight: 1 },

  // Team
  teamSec: { padding: "80px 36px", background: "#0d0d11", borderTop: "1px solid #1e1e26", display: "grid", gridTemplateColumns: "1fr 360px", gap: 60, alignItems: "center" },
  teamName: { fontFamily: "'Bebas Neue', sans-serif", fontSize: "clamp(56px,8vw,100px)", lineHeight: .88, color: "#eeeef8", letterSpacing: ".02em", marginBottom: 18 },
  teamDesc: { fontSize: 15, color: "#555568", lineHeight: 1.7, maxWidth: 440, marginBottom: 24 },
  teamStack:{ display: "flex", flexWrap: "wrap", gap: 8 },
  stackItem:{ padding: "5px 12px", border: "1px solid #2a2a36", fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568", letterSpacing: ".1em" },
  tvScore:  { background: "#111116", padding: 28, border: "1px solid #1e1e26", marginBottom: 1 },
  tvsLabel: { fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568", letterSpacing: ".18em", marginBottom: 8 },
  tvsVal:   { fontFamily: "'Bebas Neue', sans-serif", fontSize: 56, lineHeight: 1, marginBottom: 4 },
  tvsSub:   { fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568", letterSpacing: ".12em" },

  // Footer
  ftr: { padding: "20px 36px", borderTop: "1px solid #1e1e26", display: "flex", alignItems: "center", justifyContent: "space-between", fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#555568", letterSpacing: ".1em" },
};

/* ═══════════════════════════════════════════════
   TOOLTIP
═══════════════════════════════════════════════ */

function ChartTooltip({ active, payload, label, type }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#111116", border: "1px solid #1e1e26", padding: "10px 14px" }}>
      <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 9, color: "#555568", letterSpacing: ".15em", marginBottom: 6 }}>
        EPOCH {label}
      </div>
      <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 11, color: payload[0].color }}>
        {type === "loss" ? "LOSS" : "mIoU"}: {payload[0].value.toFixed(4)}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   CLASS CARD
═══════════════════════════════════════════════ */

function ClassCard({ cls, animate, delay }) {
  const pct  = Math.round(cls.iou * 100);
  const tag  = cls.iou < 0.4 ? { label: "RARE — LOW IoU", color: "#ff4d1c" }
             : cls.iou > 0.7 ? { label: "GOOD", color: "#39e075" }
             :                  { label: "OK",   color: "#555568" };

  return (
    <div style={{ ...S.classCard, borderLeft: `3px solid ${cls.color}`, transition: "background .2s" }}>
      <div style={S.ccName}>{cls.name.toUpperCase()}</div>
      <div style={S.ccTrack}>
        <div style={{
          height: "100%",
          background: cls.color,
          width: animate ? `${pct}%` : "0%",
          transition: `width 1.3s cubic-bezier(.16,1,.3,1) ${delay}ms`,
        }} />
      </div>
      <div style={S.ccRow}>
        <span style={S.ccPct}>
          {pct}<span style={{ fontSize: 14, color: "#555568" }}>%</span>
        </span>
        <span style={S.ccWeight}>weight ×{cls.weight}</span>
      </div>
      <span style={{
        display: "inline-block", marginTop: 6, marginLeft: 10,
        padding: "2px 7px", border: `1px solid ${tag.color}`,
        color: tag.color, fontFamily: "'DM Mono',monospace", fontSize: 8, letterSpacing: ".1em",
      }}>
        {tag.label}
      </span>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════ */

export default function App() {
  const miou         = useCountUp(0.3911, 1600, 800);
  const bannerMiou   = useCountUp(0.3911, 1600, 400);

  // Hero stat counts
  const stat2857 = useCountUp(2857, 1400, 600);
  const stat317  = useCountUp(317,  1200, 700);
  const stat11   = useCountUp(11,   1000, 800);
  const stat10   = useCountUp(10,   900,  900);

  // Bars animate on scroll
  const classRef  = useRef(null);
  const barsReady = useInView(classRef, 0.15);

  // Pulse for badge dot
  const [dotPulse, setDotPulse] = useState(true);
  useEffect(() => {
    const t = setInterval(() => setDotPulse((v) => !v), 900);
    return () => clearInterval(t);
  }, []);

  return (
    <>
      {/* Google Fonts */}
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet" />

      <div style={S.app}>

        {/* ── HEADER ── */}
        <header style={S.hdr}>
          <div style={S.hdrLeft}>
            <div style={S.badge}>
              <div style={{ ...S.badgeDot, opacity: dotPulse ? 1 : 0.3, transition: "opacity .4s" }} />
              LIVE
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={S.brandTeam}>BADMOSH CODERS</span>
              <span style={S.brandSep}>×</span>
              <span style={S.brandEvent}>DUALITY AI HACKATHON 2025</span>
            </div>
          </div>
          <div style={S.miouWrap}>
            <span style={S.miouLabel}>BEST mIoU</span>
            <span style={S.miouVal}>{miou.toFixed(4)}</span>
          </div>
        </header>

        {/* ── HERO ── */}
        <section style={S.hero}>
          {/* Grid background */}
          <div style={{
            position: "absolute", inset: 0, pointerEvents: "none",
            backgroundImage: "linear-gradient(#1e1e26 1px,transparent 1px),linear-gradient(90deg,#1e1e26 1px,transparent 1px)",
            backgroundSize: "64px 64px", opacity: .3,
            maskImage: "radial-gradient(ellipse 60% 80% at 30% 50%,black,transparent 75%)",
          }} />

          <div style={{ position: "relative", zIndex: 2 }}>
            <div style={S.heroTag}>
              <span style={{ color: "#ff4d1c" }}>◈</span>
              OFFROAD SEMANTIC SCENE SEGMENTATION
            </div>
            <h1 style={S.heroTitle}>
              <span style={{ display: "block", color: "#eeeef8" }}>DESERT</span>
              <span style={{ display: "block", color: "#eeeef8" }}>TERRAIN</span>
              <span style={{ display: "block", color: "#ff4d1c" }}>ANALYSIS</span>
            </h1>
            <p style={S.heroSub}>
              DINOv2 ViT-B/14 backbone with a custom 5-layer segmentation head,
              trained on 2,857 synthetic Falcon digital twin images across
              11 desert terrain classes by Team Badmosh Coders.
            </p>
            <div style={S.heroStats}>
              {[
                { val: Math.round(stat2857).toLocaleString(), label: "Train Images" },
                { val: Math.round(stat317),  label: "Val Images"  },
                { val: Math.round(stat11),   label: "Classes"     },
                { val: Math.round(stat10),   label: "Epochs"      },
              ].map((s, i) => (
                <div key={i} style={{ padding: "0 24px", borderRight: i < 3 ? "1px solid #2a2a36" : "none" }}>
                  <span style={S.hstatNum}>{s.val}</span>
                  <span style={S.hstatLabel}>{s.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Terrain art panel */}
          <div style={{ position: "relative", zIndex: 2, border: "1px solid #2a2a36", background: "#111116", overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(20,1fr)", gap: 1, padding: 1 }}>
              {Array.from({ length: 20 * 16 }).map((_, i) => {
                const col = i % 20, row = Math.floor(i / 20);
                const n = Math.sin(col * 0.4 + 1.2) * 0.5 + Math.sin(row * 0.35 + 0.8) * 0.5;
                const yr = row / 16;
                const cls = yr < 0.12 ? 10 : yr > 0.82 ? 9 : n > 0.6 ? 1 : n > 0.3 ? 2 : n > 0 ? 3 : n > -0.3 ? 4 : 9;
                return (
                  <div key={i} style={{ height: 18, background: CLASS_DATA[cls]?.color ?? "#1e1e26", opacity: 0.85 }} />
                );
              })}
            </div>
            <div style={{ padding: "10px 14px", fontFamily: "'DM Mono',monospace", fontSize: 9, color: "#555568", letterSpacing: ".18em" }}>
              SYNTHETIC DESERT TWIN — FALCON PLATFORM
            </div>
          </div>
        </section>

        {/* ── BANNER ── */}
        <div style={S.banner}>
          {[
            { label: "Best Val mIoU", val: bannerMiou.toFixed(4), color: "#ff4d1c", sub: "↑ TARGET 0.90" },
            { label: "Backbone",       val: "ViT-B",               color: "#eeeef8", sub: "768-DIM EMBEDDINGS" },
            { label: "Resolution",     val: "952", unit: "×532",    color: "#eeeef8", sub: "FULL RES INPUT" },
            { label: "Parameters",     val: "92",  unit: "M",       color: "#eeeef8", sub: "86M BACKBONE + 5.6M HEAD" },
            { label: "Report Score",   val: "20",  unit: "pts",     color: "#39e075", sub: "DOCUMENTATION" },
          ].map((item, i) => (
            <div key={i} style={S.bannerItem}>
              <div style={S.biLabel}>{item.label}</div>
              <div style={{ ...S.biVal, color: item.color }}>
                {item.val}
                {item.unit && <span style={S.biUnit}>{item.unit}</span>}
              </div>
              <div style={S.biSub}>{item.sub}</div>
            </div>
          ))}
        </div>

        {/* ── CLASS BREAKDOWN ── */}
        <section style={{ ...S.section, borderTop: "1px solid #1e1e26" }} ref={classRef}>
          <div style={S.secEyebrow}>// 03 — CLASS PERFORMANCE</div>
          <div style={S.secTitle}>Per-Class IoU</div>
          <div style={S.classGrid}>
            {CLASS_DATA.map((cls, i) => (
              <ClassCard key={cls.name} cls={cls} animate={barsReady} delay={i * 60} />
            ))}
          </div>
        </section>

        {/* ── CHARTS ── */}
        <section style={{ ...S.section, background: "#0d0d11", borderTop: "1px solid #1e1e26" }}>
          <div style={S.secEyebrow}>// 04 — TRAINING CURVES</div>
          <div style={S.secTitle}>Performance Over Time</div>
          <div style={S.chartsGrid}>
            {/* Loss */}
            <div style={S.chartCard}>
              <div style={S.ccTitle}>TRAINING LOSS <span style={{ color: "#555568", marginLeft: 10, fontSize: 9 }}>CE + DICE</span></div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={TRAINING_DATA} margin={{ top: 4, right: 4, left: -18, bottom: 0 }}>
                  <defs>
                    <linearGradient id="lG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#ff4d1c" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#ff4d1c" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e26" vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fill:"#555568", fontFamily:"'DM Mono',monospace", fontSize:9 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill:"#555568", fontFamily:"'DM Mono',monospace", fontSize:9 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<ChartTooltip type="loss" />} />
                  <Area type="monotone" dataKey="loss" stroke="#ff4d1c" strokeWidth={2} fill="url(#lG)" dot={{ fill:"#ff4d1c", r:3, strokeWidth:2, stroke:"#060608" }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* IoU */}
            <div style={S.chartCard}>
              <div style={S.ccTitle}>VALIDATION mIoU <span style={{ color: "#555568", marginLeft: 10, fontSize: 9 }}>10 EPOCHS</span></div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={TRAINING_DATA} margin={{ top: 4, right: 4, left: -18, bottom: 0 }}>
                  <defs>
                    <linearGradient id="iG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#39e075" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#39e075" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e26" vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fill:"#555568", fontFamily:"'DM Mono',monospace", fontSize:9 }} tickLine={false} axisLine={false} />
                  <YAxis domain={[0,1]} tick={{ fill:"#555568", fontFamily:"'DM Mono',monospace", fontSize:9 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<ChartTooltip type="iou" />} />
                  <Area type="monotone" dataKey="iou" stroke="#39e075" strokeWidth={2} fill="url(#iG)" dot={{ fill:"#39e075", r:3, strokeWidth:2, stroke:"#060608" }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* ── IMPROVEMENTS ── */}
        <section style={{ ...S.section, borderTop: "1px solid #1e1e26" }}>
          <div style={S.secEyebrow}>// 05 — OPTIMIZATIONS</div>
          <div style={S.secTitle}>What We Improved</div>
          <div style={S.impGrid}>
            {IMPROVEMENTS.map((imp) => (
              <div key={imp.num} style={S.impCard}>
                <div style={S.impTitle}>{imp.title}</div>
                <div style={S.impDesc}>{imp.desc}</div>
                <div style={S.impTag}>{imp.tag}</div>
                <div style={S.impNum}>{imp.num}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── TEAM ── */}
        <section style={S.teamSec}>
          <div>
            <div style={S.secEyebrow}>// 06 — THE TEAM</div>
            <h2 style={S.teamName}>
              BADMOSH<br />CODERS
            </h2>
            <p style={S.teamDesc}>
              Building bold AI solutions with zero chill and maximum precision.
              Trained on synthetic desert data. Deployed on real ambition.
              Duality AI Hackathon 2025.
            </p>
            <div style={S.teamStack}>
              {["PyTorch", "DINOv2", "Google Colab", "Falcon Platform", "Duality AI"].map((t) => (
                <span key={t} style={S.stackItem}>{t}</span>
              ))}
            </div>
          </div>
          <div>
            {[
              { label: "FINAL mIoU SCORE",   val: "0.3911", color: "#ff4d1c", sub: "VALIDATION SET"       },
              { label: "HACKATHON POINTS",    val: "100",    color: "#39e075", sub: "80 MODEL + 20 DOCS"  },
            ].map((s) => (
              <div key={s.label} style={S.tvScore}>
                <div style={S.tvsLabel}>{s.label}</div>
                <div style={{ ...S.tvsVal, color: s.color }}>{s.val}</div>
                <div style={S.tvsSub}>{s.sub}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── FOOTER ── */}
        <footer style={S.ftr}>
          <span>
            <span style={{ color: "#eeeef8", fontWeight: 500 }}>BADMOSH CODERS</span>
            <span style={{ margin: "0 10px", color: "#ff4d1c" }}>·</span>
            DUALITY AI HACKATHON 2025
          </span>
          <span>DINOv2 ViT-B/14 · 11 CLASSES · SYNTHETIC DATA · FALCON PLATFORM</span>
        </footer>

      </div>
    </>
  );
}
