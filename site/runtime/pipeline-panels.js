/**
 * pipeline-panels.js — Dynamic rendering of Bonfyre pipeline outputs
 *
 * Loads real JSON/text from pipeline artifacts and renders interactive
 * panels: transcript viewer, quality metrics, tone bars, repurpose
 * previews, pipeline timing waterfall, offer pricing, tag clouds.
 *
 * Usage:
 *   BonfyrePanels.init({ basePath: 'demos/app/proofs/slug', panels: [...] });
 *   BonfyrePanels.renderCard(container, item, panelConfig);
 */
(function(global) {
  'use strict';

  /* ── Tiny helpers ─────────────────────────────────────────── */
  function esc(v) { return String(v==null?'':v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]); }
  function pct(v,max) { return Math.min(100,Math.max(0,((v||0)/max)*100)); }
  function fmt(n,d) { return Number(n||0).toFixed(d===undefined?1:d); }
  function dur(s) { var m=Math.floor(s/60); return m?m+'m '+Math.round(s%60)+'s':fmt(s,1)+'s'; }

  async function load(url) {
    try {
      var r = await fetch(url + '?v=' + Date.now(), { cache: 'no-store' });
      if (!r.ok) return null;
      var ct = r.headers.get('content-type') || '';
      if (ct.indexOf('json') >= 0) return r.json();
      var t = await r.text();
      try { return JSON.parse(t); } catch(_) { return t; }
    } catch(_) { return null; }
  }

  async function loadText(url) {
    try {
      var r = await fetch(url + '?v=' + Date.now(), { cache: 'no-store' });
      return r.ok ? r.text() : null;
    } catch(_) { return null; }
  }

  /* ── Style injection ──────────────────────────────────────── */
  function injectStyles() {
    if (document.getElementById('bfp-styles')) return;
    var s = document.createElement('style');
    s.id = 'bfp-styles';
    s.textContent = `
.bfp-panels{display:grid;gap:.6rem;margin-top:.85rem;}
.bfp-panel{border:1px solid #1e293b;border-radius:10px;background:#0c1220;overflow:hidden;transition:all .2s;}
.bfp-panel.open .bfp-body{display:block;}
.bfp-head{display:flex;align-items:center;gap:.6rem;padding:.65rem .85rem;cursor:pointer;user-select:none;}
.bfp-head:hover{background:rgba(255,102,0,.04);}
.bfp-icon{font-size:.95rem;width:1.4rem;text-align:center;flex-shrink:0;}
.bfp-title{flex:1;font-size:.78rem;font-weight:600;color:#e2e8f0;letter-spacing:.02em;}
.bfp-badge{font-size:.65rem;padding:2px 7px;border-radius:999px;font-weight:600;white-space:nowrap;}
.bfp-badge-green{background:#052e16;color:#4ade80;border:1px solid #166534;}
.bfp-badge-amber{background:#422006;color:#fbbf24;border:1px solid #92400e;}
.bfp-badge-red{background:#450a0a;color:#f87171;border:1px solid #991b1b;}
.bfp-badge-blue{background:#0c1a3d;color:#60a5fa;border:1px solid #1e3a8a;}
.bfp-badge-purple{background:#1e103d;color:#a78bfa;border:1px solid #5b21b6;}
.bfp-chevron{font-size:.7rem;color:#64748b;transition:transform .15s;}
.bfp-panel.open .bfp-chevron{transform:rotate(90deg);}
.bfp-body{display:none;padding:.65rem .85rem .85rem;border-top:1px solid #1e293b;}

/* Tone bars */
.bfp-bar-row{display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem;}
.bfp-bar-label{font-size:.72rem;color:#94a3b8;width:70px;flex-shrink:0;text-align:right;}
.bfp-bar-track{flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden;position:relative;}
.bfp-bar-fill{height:100%;border-radius:4px;transition:width .6s ease;}

/* Transcript */
.bfp-transcript{max-height:260px;overflow-y:auto;font-size:.8rem;line-height:1.55;color:#cbd5e1;padding:.6rem;background:#0a0f1a;border-radius:6px;white-space:pre-wrap;word-break:break-word;}
.bfp-transcript-toggle{display:flex;gap:.5rem;margin-bottom:.5rem;}
.bfp-toggle-btn{font-size:.7rem;padding:3px 10px;border-radius:4px;border:1px solid #334155;background:transparent;color:#94a3b8;cursor:pointer;}
.bfp-toggle-btn.active{background:#1e3a5f;color:#60a5fa;border-color:#2563eb;}

/* Repurpose tabs */
.bfp-tabs{display:flex;gap:0;border-bottom:1px solid #1e293b;margin-bottom:.6rem;}
.bfp-tab{font-size:.72rem;padding:.4rem .75rem;cursor:pointer;color:#64748b;border-bottom:2px solid transparent;transition:all .15s;}
.bfp-tab:hover{color:#94a3b8;}
.bfp-tab.active{color:#ff6600;border-color:#ff6600;}
.bfp-tab-content{font-size:.8rem;line-height:1.55;color:#cbd5e1;white-space:pre-wrap;max-height:280px;overflow-y:auto;padding:.5rem;background:#0a0f1a;border-radius:6px;}

/* Pipeline waterfall */
.bfp-waterfall{display:grid;gap:3px;}
.bfp-wf-row{display:flex;align-items:center;gap:.4rem;font-size:.68rem;}
.bfp-wf-name{width:90px;text-align:right;color:#64748b;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.bfp-wf-bar{height:12px;border-radius:3px;min-width:2px;transition:width .4s ease;}
.bfp-wf-time{color:#94a3b8;width:45px;flex-shrink:0;}

/* Tags */
.bfp-tag-cloud{display:flex;gap:.4rem;flex-wrap:wrap;}
.bfp-tag{font-size:.7rem;padding:3px 9px;border-radius:999px;border:1px solid #334155;background:#111827;color:#cbd5e1;cursor:default;transition:all .15s;}
.bfp-tag:hover{border-color:#ff6600;color:#ff6600;}

/* Quality grid */
.bfp-qgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:.5rem;}
.bfp-qbox{text-align:center;padding:.55rem;border:1px solid #1e293b;border-radius:8px;background:#0a101c;}
.bfp-qnum{font-size:1.1rem;font-weight:700;color:#e2e8f0;}
.bfp-qlabel{font-size:.65rem;color:#64748b;margin-top:.15rem;}

/* Offer card */
.bfp-offer{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;}
.bfp-offer-main{padding:.65rem;border:1px solid #1e3a2a;border-radius:8px;background:#061210;}
.bfp-offer-price{font-size:1.6rem;font-weight:800;color:#4ade80;}
.bfp-offer-label{font-size:.7rem;color:#64748b;}
.bfp-offer-detail{font-size:.78rem;color:#94a3b8;margin-top:.25rem;}

/* Markdown renderer */
.bfp-md h2{font-size:.9rem;color:#ff6600;margin:.6rem 0 .3rem;}
.bfp-md h3{font-size:.82rem;color:#93c5fd;margin:.5rem 0 .2rem;}
.bfp-md ul{margin:.2rem 0 .4rem 1.2rem;font-size:.8rem;color:#cbd5e1;line-height:1.5;}
.bfp-md p{font-size:.8rem;color:#cbd5e1;line-height:1.5;margin:.15rem 0;}
.bfp-md strong{color:#e2e8f0;}

/* Segment confidence heatmap */
.bfp-heatmap{display:flex;gap:2px;flex-wrap:wrap;margin-top:.4rem;}
.bfp-heat-seg{width:14px;height:14px;border-radius:2px;cursor:pointer;transition:transform .1s;position:relative;}
.bfp-heat-seg:hover{transform:scale(1.5);z-index:1;}
.bfp-heat-tooltip{display:none;position:absolute;bottom:120%;left:50%;transform:translateX(-50%);background:#1e293b;color:#e2e8f0;padding:4px 8px;border-radius:4px;font-size:.65rem;white-space:nowrap;z-index:10;pointer-events:none;}
.bfp-heat-seg:hover .bfp-heat-tooltip{display:block;}

/* Source verification banner */
.bfp-source-banner{margin-bottom:.65rem;border:1px solid #1e3a2a;border-radius:8px;background:#061210;overflow:hidden;}
.bfp-source-play{display:flex;align-items:center;gap:.75rem;padding:.6rem .85rem;text-decoration:none;color:#e2e8f0;transition:background .15s;}
.bfp-source-play:hover{background:rgba(255,102,0,.06);}
.bfp-play-icon{font-size:1.3rem;width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,102,0,.12);color:#ff6600;flex-shrink:0;transition:transform .15s;}
.bfp-source-play:hover .bfp-play-icon{transform:scale(1.08);}
.bfp-source-info{display:flex;flex-direction:column;min-width:0;}
.bfp-source-title{font-size:.82rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.bfp-source-sub{font-size:.68rem;color:#4ade80;}

/* Timestamped transcript segments */
.bfp-timed-transcript{padding:0!important;}
.bfp-seg-row{display:flex;align-items:flex-start;gap:.5rem;padding:.35rem .6rem;border-bottom:1px solid #0f172a;font-size:.8rem;transition:background .1s;}
.bfp-seg-row:hover{background:rgba(255,102,0,.04);}
.bfp-seg-row.bfp-seg-hall{background:rgba(239,68,68,.06);border-left:2px solid #ef4444;}
.bfp-seg-row.bfp-seg-low{background:rgba(249,115,22,.04);border-left:2px solid #f97316;}
.bfp-ts-link{text-decoration:none;flex-shrink:0;}
.bfp-ts-link:hover .bfp-ts-time{color:#ff6600;background:rgba(255,102,0,.15);}
.bfp-ts-nolink{flex-shrink:0;}
.bfp-ts-time{display:inline-block;font-size:.68rem;font-family:monospace;color:#60a5fa;background:#0c1a3d;padding:1px 6px;border-radius:3px;min-width:42px;text-align:center;transition:all .12s;white-space:nowrap;}
.bfp-seg-text{flex:1;color:#cbd5e1;line-height:1.45;min-width:0;}
.bfp-seg-conf{font-size:.65rem;font-weight:600;flex-shrink:0;min-width:32px;text-align:right;}
`;
    document.head.appendChild(s);
  }

  /* ── Panel Renderers ──────────────────────────────────────── */

  function renderQuality(data) {
    if (!data || !data.transcribe) return '';
    var t = data.transcribe;
    var conf = t.avg_confidence || t.avg_conf || 0;
    var segs = t.segments_total || 0;
    var hall = t.segments_hallucinated || 0;
    var hallPct = segs ? Math.round((hall / segs) * 100) : 0;
    var rtf = t.rtf || 0;
    var confPct = Math.round(conf * 100);
    var confColor = conf >= 0.8 ? '#4ade80' : conf >= 0.5 ? '#fbbf24' : '#f87171';
    var dashOffset = 283 - (283 * conf);
    var html = '<div style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;">';
    // SVG confidence ring
    html += '<div style="position:relative;width:80px;height:80px;flex-shrink:0;">';
    html += '<svg viewBox="0 0 100 100" style="width:80px;height:80px;transform:rotate(-90deg)">';
    html += '<circle cx="50" cy="50" r="45" fill="none" stroke="#1e293b" stroke-width="8"/>';
    html += '<circle cx="50" cy="50" r="45" fill="none" stroke="' + confColor + '" stroke-width="8" stroke-linecap="round" stroke-dasharray="283" stroke-dashoffset="' + dashOffset + '" style="transition:stroke-dashoffset .8s ease"/>';
    html += '</svg>';
    html += '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;">';
    html += '<div style="font-size:1.1rem;font-weight:800;color:' + confColor + '">' + confPct + '%</div>';
    html += '<div style="font-size:.55rem;color:#64748b;">confidence</div>';
    html += '</div></div>';
    // Metrics grid
    html += '<div class="bfp-qgrid" style="flex:1;min-width:200px;">';
    html += '<div class="bfp-qbox"><div class="bfp-qnum">' + segs + '</div><div class="bfp-qlabel">Segments</div></div>';
    html += '<div class="bfp-qbox"><div class="bfp-qnum" style="color:' +
      (hallPct <= 5 ? '#4ade80' : hallPct <= 20 ? '#fbbf24' : '#f87171') + '">' +
      hallPct + '%</div><div class="bfp-qlabel">Hallucinated</div></div>';
    if (rtf) html += '<div class="bfp-qbox"><div class="bfp-qnum">' + fmt(rtf, 3) + '</div><div class="bfp-qlabel">Realtime Factor</div></div>';
    var durSec = (data.source || {}).duration_seconds;
    if (durSec) html += '<div class="bfp-qbox"><div class="bfp-qnum">' + dur(durSec) + '</div><div class="bfp-qlabel">Duration</div></div>';
    html += '</div></div>';
    return html;
  }

  function renderSegmentHeatmap(segments, sourceUrl) {
    if (!segments || !segments.length) return '';
    var html = '<div style="margin-top:.5rem;"><div style="font-size:.68rem;color:#64748b;margin-bottom:.3rem;">Segment confidence heatmap — ' + (sourceUrl ? 'click to verify at timestamp' : 'hover for details') + '</div><div class="bfp-heatmap">';
    for (var i = 0; i < segments.length; i++) {
      var s = segments[i];
      var c = s.confidence || 0;
      var hcp = s.hcp_quality;
      var hall = s.hallucination_flags || 0;
      var color = hall > 0 ? '#ef4444' : c >= 0.85 ? '#22c55e' : c >= 0.6 ? '#eab308' : '#f97316';
      var t0s = Math.floor((s.t0_ms || 0) / 1000);
      var tip = 'Seg ' + (i + 1) + ' [' + dur(t0s) + ']: ' + fmt(c * 100, 0) + '% conf';
      if (hcp !== undefined && hcp > 0) tip += ', HCP ' + fmt(hcp * 100, 0) + '%';
      if (hall > 0) tip += ' ⚠ hallucinated';
      var linkAttr = sourceUrl && s.t0_ms !== undefined ? ' data-src-url="' + esc(sourceUrl) + '" data-t0="' + t0s + '" style="background:' + color + ';cursor:pointer;"' : ' style="background:' + color + '"';
      html += '<div class="bfp-heat-seg"' + linkAttr + '><div class="bfp-heat-tooltip">' + esc(tip) + (sourceUrl ? '<br>▸ click to verify' : '') + '</div></div>';
    }
    html += '</div></div>';
    return html;
  }

  function renderTone(data) {
    if (!data) return '';
    var feats = data.features || data;
    // Extract meaningful derived metrics from eGeMAPSv02
    var pitch = feats['F0semitoneFrom27.5Hz_sma3nz_amean'] || 0;
    var pitchVar = feats['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] || 0;
    var loudness = feats['loudness_sma3_amean'] || 0;
    var loudVar = feats['loudness_sma3_stddevNorm'] || 0;
    var spectralFlux = feats['spectralFlux_sma3_amean'] || 0;
    var hnr = feats['HNRdBACF_sma3nz_amean'] || 0;
    var shimmer = feats['shimmerLocaldB_sma3nz_amean'] || 0;
    var jitter = feats['jitterLocal_sma3nz_amean'] || 0;
    // Derive intuitive metrics (normalized 0-100)
    var energy = Math.min(100, (loudness / 0.8) * 100);
    var expressiveness = Math.min(100, pitchVar * 250);
    var clarity = Math.min(100, Math.max(0, (hnr / 20) * 100));
    var dynamicRange = Math.min(100, loudVar * 80);
    var stability = Math.min(100, Math.max(0, 100 - jitter * 2000));
    var bars = [
      { label: 'Energy', value: energy, color: '#4ade80' },
      { label: 'Expressiveness', value: expressiveness, color: '#fbbf24' },
      { label: 'Clarity', value: clarity, color: '#60a5fa' },
      { label: 'Dynamics', value: dynamicRange, color: '#818cf8' },
      { label: 'Stability', value: stability, color: '#a78bfa' }
    ];
    // Also show raw pitch
    if (pitch > 0) bars.push({ label: 'Pitch', value: Math.min(100, (pitch / 60) * 100), color: '#f97316' });
    var html = '';
    for (var j = 0; j < bars.length; j++) {
      html += '<div class="bfp-bar-row"><div class="bfp-bar-label">' + esc(bars[j].label) + '</div>' +
        '<div class="bfp-bar-track"><div class="bfp-bar-fill" style="width:' + pct(bars[j].value, 100) +
        '%;background:' + bars[j].color + '"></div></div>' +
        '<span style="font-size:.65rem;color:#64748b;width:35px;text-align:right;">' + fmt(bars[j].value, 0) + '</span></div>';
    }
    return html;
  }

  function renderToneCSV(csv) {
    if (!csv) return '';
    var lines = csv.trim().split('\n');
    if (lines.length < 2) return '';
    var headers = lines[0].split(';');
    var values = lines[1].split(';');
    var features = [];
    for (var i = 1; i < headers.length && i < values.length; i++) {
      var v = parseFloat(values[i]);
      if (!isNaN(v) && headers[i]) features.push({ name: headers[i].replace(/^.*_/, ''), value: v });
    }
    features.sort(function(a, b) { return Math.abs(b.value) - Math.abs(a.value); });
    features = features.slice(0, 8);
    if (!features.length) return '';
    var maxVal = Math.max.apply(null, features.map(function(f) { return Math.abs(f.value); }));
    var html = '<div style="font-size:.68rem;color:#64748b;margin-bottom:.3rem;">Top speech features (OpenSMILE)</div>';
    for (var j = 0; j < features.length; j++) {
      var f = features[j];
      var norm = maxVal > 0 ? (Math.abs(f.value) / maxVal) * 100 : 0;
      html += '<div class="bfp-bar-row"><div class="bfp-bar-label">' + esc(f.name) + '</div>' +
        '<div class="bfp-bar-track"><div class="bfp-bar-fill" style="width:' + norm +
        '%;background:#818cf8"></div></div></div>';
    }
    return html;
  }

  function mdToHtml(text) {
    return esc(String(text || ''))
      .replace(/^### (.*)$/gm, '<h3>$1</h3>')
      .replace(/^## (.*)$/gm, '<h2>$1</h2>')
      .replace(/^# (.*)$/gm, '<h2>$1</h2>')
      .replace(/^\*\*(\d+\/)\*\*$/gm, '<strong style="color:#ff6600">$1</strong>')
      .replace(/^\- (.*)$/gm, '<li>$1</li>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/→/g, '<span style="color:#ff6600">→</span>')
      .replace(/\n\n+/g, '<br><br>')
      .replace(/\n/g, '<br>');
  }

  function renderRepurpose(tweets, linkedin, youtube) {
    var tabs = [];
    if (tweets) tabs.push({ id: 'tweet', label: '𝕏 Tweet Thread', content: tweets });
    if (linkedin) tabs.push({ id: 'linkedin', label: 'LinkedIn', content: linkedin });
    if (youtube) tabs.push({ id: 'youtube', label: 'YouTube Desc', content: youtube });
    if (!tabs.length) return '';
    var html = '<div class="bfp-tabs">';
    for (var i = 0; i < tabs.length; i++) {
      html += '<div class="bfp-tab' + (i === 0 ? ' active' : '') + '" data-tab="' + tabs[i].id + '">' + tabs[i].label + '</div>';
    }
    html += '</div>';
    for (var j = 0; j < tabs.length; j++) {
      html += '<div class="bfp-tab-content" data-tabcontent="' + tabs[j].id + '" style="' + (j > 0 ? 'display:none' : '') + '">' +
        '<div class="bfp-md">' + mdToHtml(tabs[j].content) + '</div></div>';
    }
    return html;
  }

  function renderWaterfall(recipe) {
    if (!recipe || !recipe.steps) return '';
    var steps = recipe.steps;
    var maxWall = 0;
    for (var i = 0; i < steps.length; i++) {
      if ((steps[i].wall_s || 0) > maxWall) maxWall = steps[i].wall_s;
    }
    if (!maxWall) return '';
    var colors = {
      'fetch-metadata': '#64748b', 'download-audio': '#64748b', 'media-prep': '#64748b',
      'transcribe': '#f97316', 'transcript-clean': '#f97316', 'paragraph': '#f97316',
      'brief': '#60a5fa', 'tag': '#a78bfa', 'tone': '#818cf8', 'embed': '#818cf8',
      'render': '#22c55e', 'hash': '#22c55e', 'offer': '#fbbf24', 'pack': '#fbbf24',
      'repurpose-tweet': '#4ade80', 'repurpose-linkedin': '#4ade80', 'repurpose-youtube': '#4ade80',
      'emit-html': '#4ade80', 'index': '#4ade80', 'compress': '#94a3b8',
      'stitch': '#94a3b8', 'ledger': '#94a3b8', 'meter': '#94a3b8'
    };
    var html = '<div class="bfp-waterfall">';
    for (var j = 0; j < steps.length; j++) {
      var s = steps[j];
      if (s.exit !== 0 && s.exit !== undefined) continue;
      var w = maxWall > 0 ? (s.wall_s / maxWall) * 100 : 0;
      var c = colors[s.name] || '#64748b';
      html += '<div class="bfp-wf-row">' +
        '<div class="bfp-wf-name">' + esc(s.name) + '</div>' +
        '<div class="bfp-wf-bar" style="width:' + Math.max(2, w) + '%;background:' + c + '"></div>' +
        '<div class="bfp-wf-time">' + dur(s.wall_s || 0) + '</div></div>';
    }
    html += '</div>';
    if (recipe.total_s) {
      html += '<div style="margin-top:.5rem;font-size:.72rem;color:#94a3b8;">Total: <strong style="color:#e2e8f0;">' + dur(recipe.total_s) + '</strong>';
      if (recipe.source && recipe.source.duration_seconds) {
        html += ' for ' + dur(recipe.source.duration_seconds) + ' of audio';
      }
      html += '</div>';
    }
    return html;
  }

  function renderOffer(data) {
    if (!data) return '';
    var price = data.price || ('$' + fmt(data.priceCents ? data.priceCents / 100 : (parseFloat(String(data.price_usd||'0').replace(/[^0-9.]/g,'')) || 0), 2));
    var turnaround = data.turnaround || data.delivery || 'same-day';
    var headline = data.headline || '';
    var promise = data.promise || '';
    var pf = data.pricingFactors;
    var html = '<div class="bfp-offer">';
    html += '<div class="bfp-offer-main"><div class="bfp-offer-price">' + esc(price) + '</div>' +
      '<div class="bfp-offer-label">Estimated Value</div>';
    if (headline) html += '<div class="bfp-offer-detail" style="margin-top:.4rem;color:#e2e8f0;font-weight:600;">' + esc(headline) + '</div>';
    html += '</div>';
    html += '<div class="bfp-offer-main"><div class="bfp-offer-price" style="font-size:1.1rem;color:#60a5fa;">' +
      esc(turnaround) + '</div><div class="bfp-offer-label">Turnaround</div>';
    if (promise) html += '<div class="bfp-offer-detail" style="margin-top:.4rem;">' + esc(promise) + '</div>';
    if (pf) {
      html += '<div style="margin-top:.4rem;font-size:.68rem;color:#64748b;">';
      html += 'Base: $' + fmt(pf.base, 2);
      if (pf.wordPremium) html += ' + word: $' + fmt(pf.wordPremium, 2);
      if (pf.qualityMult && pf.qualityMult !== 1) html += ' × quality: ' + fmt(pf.qualityMult, 1);
      html += '</div>';
    }
    html += '</div></div>';
    return html;
  }

  function renderLedger(data) {
    if (!data) return '';
    var html = '<div class="bfp-qgrid">';
    var keys = Object.keys(data);
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var v = data[k];
      if (typeof v === 'number') {
        html += '<div class="bfp-qbox"><div class="bfp-qnum">' + (v >= 100 ? Math.round(v) : fmt(v, 2)) +
          '</div><div class="bfp-qlabel">' + esc(k.replace(/_/g, ' ')) + '</div></div>';
      } else if (typeof v === 'string') {
        html += '<div class="bfp-qbox"><div class="bfp-qnum" style="font-size:.85rem;">' + esc(v) +
          '</div><div class="bfp-qlabel">' + esc(k.replace(/_/g, ' ')) + '</div></div>';
      }
    }
    html += '</div>';
    return html;
  }

  function renderTags(tags) {
    if (!tags || !tags.length) return '';
    var html = '<div class="bfp-tag-cloud">';
    for (var i = 0; i < tags.length; i++) {
      html += '<div class="bfp-tag">' + esc(tags[i]) + '</div>';
    }
    html += '</div>';
    return html;
  }

  /* ── Panel builder ────────────────────────────────────────── */

  var PANEL_DEFS = {
    quality: {
      icon: '📊', title: 'Quality Metrics',
      badgeFn: function(d) {
        var c = (d.transcribe||{}).avg_confidence || 0;
        return { text: fmt(c*100,0)+'%', cls: c>=0.8?'green':c>=0.5?'amber':'red' };
      }
    },
    transcript: { icon: '📝', title: 'Clean Transcript', badge: { text: 'expand', cls: 'blue' } },
    tone: { icon: '🎭', title: 'Tone Analysis', badge: { text: 'speech', cls: 'purple' } },
    repurpose: { icon: '📱', title: 'Social Media Formats', badge: { text: 'ready', cls: 'green' } },
    pipeline: { icon: '⚡', title: 'Pipeline Timing', badge: { text: 'waterfall', cls: 'blue' } },
    offer: { icon: '💰', title: 'Pricing', badgeFn: function(d) {
      var p = d.priceCents ? d.priceCents/100 : (parseFloat(String(d.price_usd||d.price||'0').replace(/[^0-9.]/g,'')) || 0);
      return { text: '$'+fmt(p,2), cls: 'green' };
    } },
    tags: { icon: '🏷️', title: 'Topics', badge: { text: 'auto', cls: 'purple' } },
    brief: { icon: '📋', title: 'Brief', badge: { text: 'summary', cls: 'blue' } },
    ledger: { icon: '📒', title: 'Ledger', badge: { text: 'value', cls: 'amber' } },
    segments: { icon: '🧬', title: 'Segment Heatmap', badge: { text: 'confidence', cls: 'blue' } }
  };

  function buildPanel(id, def, content, data) {
    var badge = def.badge || (def.badgeFn ? def.badgeFn(data) : { text: '', cls: 'blue' });
    return '<div class="bfp-panel" data-panel="' + id + '">' +
      '<div class="bfp-head" onclick="BonfyrePanels._toggle(this)">' +
      '<div class="bfp-icon">' + def.icon + '</div>' +
      '<div class="bfp-title">' + esc(def.title) + '</div>' +
      (badge.text ? '<div class="bfp-badge bfp-badge-' + badge.cls + '">' + esc(badge.text) + '</div>' : '') +
      '<div class="bfp-chevron">▸</div>' +
      '</div>' +
      '<div class="bfp-body">' + content + '</div></div>';
  }

  /* ── Main public API ──────────────────────────────────────── */

  function toggle(headEl) {
    var panel = headEl.closest('.bfp-panel');
    if (panel) panel.classList.toggle('open');
  }

  function bindTabs(container) {
    if (!container) return;
    container.addEventListener('click', function(e) {
      var tab = e.target.closest('.bfp-tab');
      if (!tab) return;
      var parent = tab.closest('.bfp-body') || tab.closest('.bfp-panel');
      if (!parent) return;
      var tabs = parent.querySelectorAll('.bfp-tab');
      var contents = parent.querySelectorAll('.bfp-tab-content');
      for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
      for (var j = 0; j < contents.length; j++) contents[j].style.display = 'none';
      tab.classList.add('active');
      var target = parent.querySelector('[data-tabcontent="' + tab.dataset.tab + '"]');
      if (target) target.style.display = '';
    });
  }

  /**
   * renderPanels — Load pipeline artifacts and render all panels into a container
   *
   * @param {Element} container - DOM element to render into
   * @param {Object} opts - { basePath, panels, proofData, recipeData, toneData, ... }
   */
  async function renderPanels(container, opts) {
    injectStyles();
    if (!container) return;

    var base = opts.basePath || '';
    var panels = opts.panels || ['quality', 'transcript', 'tone', 'repurpose', 'pipeline', 'offer', 'tags'];

    // Try nested path first, fallback to flat
    function loadFallback(a, b) { return load(a).then(function(r) { return r || load(b); }); }
    function loadTextFallback(a, b) { return loadText(a).then(function(r) { return r || loadText(b); }); }

    // Load all artifacts in parallel
    var loads = {};
    if (panels.indexOf('quality') >= 0 || panels.indexOf('segments') >= 0 || panels.indexOf('transcript') >= 0) {
      loads.proof = opts.proofData || load(base + '/proof.json');
      loads.transcript = loadFallback(base + '/transcribe/transcript.json', base + '/transcript.json');
    }
    if (panels.indexOf('transcript') >= 0) {
      loads.clean = loadTextFallback(base + '/clean/clean.txt', base + '/clean.txt');
      loads.raw = loadTextFallback(base + '/transcribe/transcript.txt', base + '/transcript.txt');
    }
    if (panels.indexOf('tone') >= 0) {
      loads.tone = opts.toneData || load(base + '/tone/tone.json');
      loads.toneCSV = loadText(base + '/tone/tone-raw.csv');
    }
    if (panels.indexOf('repurpose') >= 0) {
      loads.tweet = loadText(base + '/repurpose/tweet/tweet-thread.md');
      loads.linkedin = loadText(base + '/repurpose/linkedin/linkedin.md');
      loads.youtube = loadText(base + '/repurpose/youtube/youtube-desc.md');
    }
    if (panels.indexOf('pipeline') >= 0) {
      loads.recipe = opts.recipeData || load(base + '/recipe.json');
    }
    if (panels.indexOf('offer') >= 0) {
      loads.offer = opts.offerData || load(base + '/offer/offer.json');
    }
    if (panels.indexOf('tags') >= 0) {
      loads.tags = load(base + '/tag/lang.json');
    }
    if (panels.indexOf('brief') >= 0) {
      loads.brief = loadTextFallback(base + '/brief/brief.md', base + '/brief.md');
    }
    if (panels.indexOf('ledger') >= 0) {
      loads.ledger = load(base + '/ledger/value.json');
    }

    // Await all
    var keys = Object.keys(loads);
    var values = await Promise.all(keys.map(function(k) { return loads[k]; }));
    var d = {};
    for (var i = 0; i < keys.length; i++) d[keys[i]] = values[i];

    // Build HTML
    var html = '';

    // Source verification banner
    var srcUrl = (d.proof || {}).public_url || '';
    var srcTitle = ((d.proof || {}).source || {}).title || ((d.proof || {}).title || '');
    var srcDur = ((d.proof || {}).source || {}).duration_seconds;
    if (srcUrl) {
      html += '<div class="bfp-source-banner">';
      html += '<a class="bfp-source-play" href="' + esc(srcUrl) + '" target="_blank" rel="noreferrer" onclick="event.stopPropagation()">';
      html += '<span class="bfp-play-icon">▶</span>';
      html += '<span class="bfp-source-info">';
      html += '<span class="bfp-source-title">' + esc(srcTitle || 'Original Source') + '</span>';
      html += '<span class="bfp-source-sub">Verify against original' + (srcDur ? ' · ' + dur(srcDur) : '') + '</span>';
      html += '</span></a></div>';
    }

    html += '<div class="bfp-panels">';

    for (var p = 0; p < panels.length; p++) {
      var pid = panels[p];
      var def = PANEL_DEFS[pid];
      if (!def) continue;
      var content = '';

      switch (pid) {
        case 'quality':
          if (d.proof) content = renderQuality(d.proof);
          break;
        case 'segments':
          if (d.transcript && d.transcript.segments) content = renderSegmentHeatmap(d.transcript.segments, (d.proof || {}).public_url);
          break;
        case 'transcript':
          if (d.clean || d.raw || d.transcript) {
            var srcUrl = (d.proof || {}).public_url || '';
            content = '<div class="bfp-transcript-toggle">';
            if (d.transcript && d.transcript.segments) content += '<button class="bfp-toggle-btn active" data-view="timed">Timestamped</button>';
            content += '<button class="bfp-toggle-btn' + (d.transcript && d.transcript.segments ? '' : ' active') + '" data-view="clean">Clean</button>';
            content += '<button class="bfp-toggle-btn" data-view="raw">Raw</button></div>';
            // Timestamped view with clickable segments
            if (d.transcript && d.transcript.segments) {
              content += '<div class="bfp-transcript bfp-timed-transcript" data-transcript="timed">';
              var segs = d.transcript.segments;
              for (var si = 0; si < segs.length; si++) {
                var seg = segs[si];
                var t0s = Math.floor((seg.t0_ms || 0) / 1000);
                var t1s = Math.floor((seg.t1_ms || 0) / 1000);
                var segConf = seg.confidence || 0;
                var segHall = (seg.hallucination_flags || 0) > 0;
                var segClass = segHall ? 'bfp-seg-hall' : segConf < 0.6 ? 'bfp-seg-low' : '';
                var timeLabel = dur(t0s);
                var linkOpen = srcUrl ? '<a class="bfp-ts-link" href="' + esc(srcUrl) + (srcUrl.indexOf('youtube') >= 0 ? '&t=' + t0s : '#t=' + t0s) + '" target="_blank" rel="noreferrer" title="Verify at ' + timeLabel + '" onclick="event.stopPropagation()">' : '<span class="bfp-ts-nolink">';
                var linkClose = srcUrl ? '</a>' : '</span>';
                var segText = '';
                if (seg.text) segText = seg.text;
                else if (seg.words) segText = seg.words.map(function(w){return w.word||w.text||'';}).join(' ');
                content += '<div class="bfp-seg-row ' + segClass + '">' + linkOpen + '<span class="bfp-ts-time">' + timeLabel + '</span>' + linkClose + '<span class="bfp-seg-text">' + esc(segText) + '</span><span class="bfp-seg-conf" style="color:' + (segConf >= 0.85 ? '#4ade80' : segConf >= 0.6 ? '#fbbf24' : '#f87171') + '">' + Math.round(segConf * 100) + '%</span></div>';
              }
              content += '</div>';
            }
            content += '<div class="bfp-transcript" data-transcript="clean" style="' + (d.transcript && d.transcript.segments ? 'display:none;' : '') + '">' + esc(d.clean || 'Not available') + '</div>';
            content += '<div class="bfp-transcript" data-transcript="raw" style="display:none;">' + esc(d.raw || 'Not available') + '</div>';
          }
          break;
        case 'tone':
          if (d.tone) content = renderTone(d.tone);
          if (d.toneCSV) content += renderToneCSV(d.toneCSV);
          break;
        case 'repurpose':
          if (d.tweet || d.linkedin || d.youtube) content = renderRepurpose(d.tweet, d.linkedin, d.youtube);
          break;
        case 'pipeline':
          if (d.recipe) content = renderWaterfall(d.recipe);
          break;
        case 'offer':
          if (d.offer) content = renderOffer(d.offer);
          break;
        case 'tags':
          if (d.tags) {
            var tagList = d.tags.tags || d.tags.topics || [];
            // Handle language detection format
            if (d.tags.languages) {
              tagList = d.tags.languages.map(function(l) { return l.language + ' (' + fmt(l.confidence * 100, 0) + '%)'; });
            }
            if (typeof tagList === 'string') tagList = [tagList];
            content = renderTags(tagList);
          }
          break;
        case 'brief':
          if (d.brief) content = '<div class="bfp-md">' + mdToHtml(d.brief) + '</div>';
          break;
        case 'ledger':
          if (d.ledger) content = renderLedger(d.ledger);
          break;
      }

      if (content) html += buildPanel(pid, def, content, d[pid === 'quality' ? 'proof' : pid === 'offer' ? 'offer' : 'proof'] || {});
    }

    html += '</div>';
    container.innerHTML += html;

    // Bind interactions
    bindTabs(container);
    bindTranscriptToggle(container);
    bindHeatmapClicks(container);
  }

  function bindHeatmapClicks(container) {
    container.addEventListener('click', function(e) {
      var seg = e.target.closest('.bfp-heat-seg[data-src-url]');
      if (!seg) return;
      e.stopPropagation();
      var url = seg.dataset.srcUrl;
      var t0 = seg.dataset.t0 || '0';
      if (url) {
        var href = url + (url.indexOf('youtube') >= 0 ? '&t=' + t0 : '#t=' + t0);
        window.open(href, '_blank', 'noreferrer');
      }
    });
  }

  function bindTranscriptToggle(container) {
    container.addEventListener('click', function(e) {
      var btn = e.target.closest('.bfp-toggle-btn');
      if (!btn) return;
      var panel = btn.closest('.bfp-body');
      if (!panel) return;
      var btns = panel.querySelectorAll('.bfp-toggle-btn');
      var blocks = panel.querySelectorAll('.bfp-transcript');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      for (var j = 0; j < blocks.length; j++) blocks[j].style.display = 'none';
      btn.classList.add('active');
      var target = panel.querySelector('[data-transcript="' + btn.dataset.view + '"]');
      if (target) target.style.display = '';
    });
  }

  /* ── Auto-enhance: upgrade any Bonfyre app cards with pipeline panels ── */

  function autoEnhance(opts) {
    injectStyles();
    injectCardStyles();

    var getItems = opts.getItems;
    var boardSelector = opts.boardSelector || '#boardContent';
    var panels = opts.panels || ['quality', 'transcript', 'tone', 'repurpose', 'pipeline', 'offer'];

    // Delegate click on the board
    var board = document.querySelector(boardSelector);
    if (!board) return;

    board.addEventListener('click', function(e) {
      var card = e.target.closest('.card');
      if (!card) return;
      // Don't toggle if clicking a link/button inside
      if (e.target.closest('a') || e.target.closest('button')) return;

      var wasExpanded = card.classList.contains('bfp-expanded');
      // Collapse all
      board.querySelectorAll('.card.bfp-expanded').forEach(function(c) {
        c.classList.remove('bfp-expanded');
      });
      if (wasExpanded) return;

      card.classList.add('bfp-expanded');

      // Find or create deep container
      var deep = card.querySelector('.bfp-card-deep');
      if (!deep) {
        deep = document.createElement('div');
        deep.className = 'bfp-card-deep';
        card.appendChild(deep);
      }

      // Load panels if not yet loaded
      if (deep.dataset.loaded) return;
      deep.dataset.loaded = '1';

      // Find the item
      var itemId = card.dataset.itemId;
      var allItems = getItems ? getItems() : [];
      var item = null;
      for (var i = 0; i < allItems.length; i++) {
        if (allItems[i].id === itemId) { item = allItems[i]; break; }
      }

      if (item && item.proofPath) {
        renderPanels(deep, { basePath: item.proofPath, panels: panels }).then(function() {
          // Auto-open first panel for immediate impact
          var first = deep.querySelector('.bfp-panel');
          if (first && !first.classList.contains('open')) first.classList.add('open');
        });
      } else if (item) {
        // Build inline panels from item data even without proof files
        var html = '<div class="bfp-panels">';
        if (item.brief) {
          html += buildPanel('brief', PANEL_DEFS.brief, '<div class="bfp-md">' + mdToHtml(item.brief) + '</div>', {});
        }
        if (item.searchSummary) {
          html += buildPanel('tags', PANEL_DEFS.tags,
            '<div style="font-size:.78rem;color:#cbd5e1;line-height:1.5;">' + esc(item.searchSummary) + '</div>' +
            (item.tags ? renderTags(item.tags) : ''), {});
        }
        if (item.whyItMatters) {
          html += '<div class="bfp-panel"><div class="bfp-head" onclick="BonfyrePanels._toggle(this)">' +
            '<div class="bfp-icon">💡</div><div class="bfp-title">Why This Matters</div>' +
            '<div class="bfp-chevron">▸</div></div>' +
            '<div class="bfp-body"><div style="font-size:.82rem;color:#cbd5e1;line-height:1.55;">' +
            esc(item.whyItMatters) + '</div></div></div>';
        }
        html += '</div>';
        deep.innerHTML = html;
      }
    });

    // Patch renderBoard to add data-item-id and expand hint
    var origRender = opts.renderBoard;
    if (origRender) {
      var patchedRender = function() {
        origRender();
        // Post-render: ensure all cards have data-item-id and cursor
        var allItems = getItems ? getItems() : [];
        var cards = board.querySelectorAll('.card');
        // Build a map from item text to item for fuzzy matching
        var itemMap = {};
        for (var j = 0; j < allItems.length; j++) {
          itemMap[allItems[j].id] = allItems[j];
          if (allItems[j].file) itemMap[allItems[j].file] = allItems[j];
        }
        cards.forEach(function(card, idx) {
          if (!card.dataset.itemId) {
            // Try to match by finding item whose file name appears in card text
            var cardText = card.textContent || '';
            var matched = null;
            for (var k = 0; k < allItems.length; k++) {
              var it = allItems[k];
              if (it.file && cardText.indexOf(it.file) >= 0) { matched = it; break; }
              if (it.id && card.querySelector('[data-item-id="' + it.id + '"]')) { matched = it; break; }
            }
            if (matched) {
              card.dataset.itemId = matched.id || '';
            } else if (allItems[idx]) {
              card.dataset.itemId = allItems[idx].id || '';
            }
          }
          card.style.cursor = 'pointer';
          card.style.transition = 'border-color .2s';
          // Add expand hint if not present
          var h3 = card.querySelector('h3');
          if (h3 && !h3.querySelector('.bfp-expand-hint')) {
            var hint = document.createElement('span');
            hint.className = 'bfp-expand-hint';
            hint.textContent = 'click to explore ▸';
            h3.appendChild(hint);
          }
          // Add badge row with source link, confidence, duration
          var itemId = card.dataset.itemId;
          var item = itemMap[itemId];
          if (item && !card.querySelector('.bfp-card-badges')) {
            var badges = document.createElement('div');
            badges.className = 'bfp-card-badges';
            var bhtml = '';
            if (item.sourceUrl) {
              bhtml += '<a class="bfp-source-link" href="' + esc(item.sourceUrl) + '" target="_blank" rel="noreferrer" onclick="event.stopPropagation()">▶ Source</a>';
            }
            badges.innerHTML = bhtml;
            // Insert after card-meta
            var meta = card.querySelector('.card-meta');
            if (meta && meta.nextSibling) {
              meta.parentNode.insertBefore(badges, meta.nextSibling);
            } else {
              card.insertBefore(badges, card.querySelector('.card-tags') || card.lastChild);
            }

            if (item.proofPath) {
              // Async load proof data for rich inline metrics
              (function(badgeEl, cardEl, path) {
                var metrics = {};
                Promise.all([
                  load(path + '/proof.json'),
                  load(path + '/recipe.json'),
                  load(path + '/tone/tone.json'),
                  load(path + '/offer/offer.json')
                ]).then(function(results) {
                  var proof = results[0];
                  var recipe = results[1];
                  var tone = results[2];
                  var offer = results[3];

                  // Confidence mini-bar + number
                  if (proof && proof.transcribe) {
                    var c = proof.transcribe.avg_confidence || 0;
                    var segs = proof.transcribe.segments_total || 0;
                    var hall = proof.transcribe.segments_hallucinated || 0;
                    if (c > 0) {
                      var cls = c >= 0.8 ? 'high' : c >= 0.5 ? 'mid' : 'low';
                      var barColor = c >= 0.8 ? '#4ade80' : c >= 0.5 ? '#fbbf24' : '#f87171';
                      badgeEl.insertAdjacentHTML('beforeend',
                        '<span class="bfp-conf-badge bfp-conf-' + cls + '">' +
                        '<span class="bfp-mini-bar" style="--pct:' + Math.round(c * 100) + '%;--bar-color:' + barColor + '"></span>' +
                        Math.round(c * 100) + '%</span>');
                    }
                    var d = (proof.source || {}).duration_seconds;
                    if (d) badgeEl.insertAdjacentHTML('beforeend', '<span class="bfp-dur-badge">⏱ ' + dur(d) + '</span>');
                    if (segs > 0) badgeEl.insertAdjacentHTML('beforeend', '<span class="bfp-seg-badge">' + segs + ' segments</span>');
                  }

                  // Pipeline steps
                  if (recipe) {
                    var ok = recipe.steps_ok || (recipe.steps ? recipe.steps.filter(function(s){return s.exit===0||s.exit===undefined}).length : 0);
                    var total = recipe.steps_total || (recipe.steps ? recipe.steps.length : 0);
                    if (total > 0) badgeEl.insertAdjacentHTML('beforeend', '<span class="bfp-pipe-badge">⚡ ' + ok + '/' + total + '</span>');
                    if (recipe.total_s) badgeEl.insertAdjacentHTML('beforeend', '<span class="bfp-dur-badge">🔧 ' + dur(recipe.total_s) + ' pipeline</span>');
                  }

                  // Tone mood word
                  if (tone) {
                    var feats = tone.features || tone;
                    var loudness = feats['loudness_sma3_amean'] || 0;
                    var pitchVar = feats['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] || 0;
                    var hnr = feats['HNRdBACF_sma3nz_amean'] || 0;
                    var mood = 'Neutral';
                    var moodColor = '#94a3b8';
                    if (loudness > 0.5 && pitchVar > 0.3) { mood = 'Energetic'; moodColor = '#4ade80'; }
                    else if (loudness > 0.3 && pitchVar > 0.2) { mood = 'Expressive'; moodColor = '#fbbf24'; }
                    else if (hnr > 12) { mood = 'Clear'; moodColor = '#60a5fa'; }
                    else if (loudness < 0.2) { mood = 'Calm'; moodColor = '#a78bfa'; }
                    badgeEl.insertAdjacentHTML('beforeend',
                      '<span class="bfp-mood-badge" style="color:' + moodColor + ';border-color:' + moodColor + '30">🎭 ' + mood + '</span>');
                  }

                  // Offer price
                  if (offer) {
                    var price = offer.price || (offer.priceCents ? '$' + (offer.priceCents / 100).toFixed(2) : null) || (offer.price_usd ? '$' + Number(offer.price_usd).toFixed(2) : null);
                    if (price) badgeEl.insertAdjacentHTML('beforeend', '<span class="bfp-price-badge">💰 ' + esc(String(price)) + '</span>');
                  }

                  // Add mini metrics strip below badges
                  if (proof && proof.transcribe && proof.transcribe.avg_confidence > 0) {
                    var strip = document.createElement('div');
                    strip.className = 'bfp-metrics-strip';
                    var c = proof.transcribe.avg_confidence;
                    var hall = proof.transcribe.segments_hallucinated || 0;
                    var segs = proof.transcribe.segments_total || 0;
                    var hallPct = segs ? Math.round((hall / segs) * 100) : 0;
                    var shtml = '';
                    // Full-width confidence bar
                    var barColor = c >= 0.8 ? '#4ade80' : c >= 0.5 ? '#fbbf24' : '#f87171';
                    shtml += '<div class="bfp-strip-bar"><div class="bfp-strip-fill" style="width:' + Math.round(c * 100) + '%;background:' + barColor + '"></div></div>';
                    shtml += '<div class="bfp-strip-labels">';
                    shtml += '<span>Confidence: <strong style="color:' + barColor + '">' + Math.round(c * 100) + '%</strong></span>';
                    if (hallPct > 0) shtml += '<span>Hallucination: <strong style="color:' + (hallPct > 20 ? '#f87171' : '#fbbf24') + '">' + hallPct + '%</strong></span>';
                    if (recipe && recipe.total_s) shtml += '<span>Pipeline: <strong style="color:#60a5fa">' + dur(recipe.total_s) + '</strong></span>';
                    shtml += '</div>';
                    strip.innerHTML = shtml;
                    // Insert after badges
                    badgeEl.parentNode.insertBefore(strip, badgeEl.nextSibling);
                  }
                });
              })(badges, card, item.proofPath);
            }
          }
        });
      };
      // Return the patched function so caller can wire it into DemoBoot
      return patchedRender;
    }
  }

  function injectCardStyles() {
    if (document.getElementById('bfp-card-styles')) return;
    var s = document.createElement('style');
    s.id = 'bfp-card-styles';
    s.textContent = `
.card:hover{border-color:rgba(255,102,0,.3);}
.card.bfp-expanded{border-color:#ff6600!important;}
.bfp-card-deep{margin-top:1rem;padding-top:.85rem;border-top:1px solid #333;max-height:0;overflow:hidden;opacity:0;transition:max-height .4s ease,opacity .3s ease,padding .3s ease;padding-top:0;}
.card.bfp-expanded .bfp-card-deep{max-height:3000px;opacity:1;padding-top:.85rem;}
.bfp-expand-hint{font-size:.6rem;color:#555;margin-left:auto;font-weight:400;transition:opacity .2s;}
.card.bfp-expanded .bfp-expand-hint{opacity:0;}
.bfp-card-badges{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.5rem;}
.bfp-source-link{display:inline-flex;align-items:center;gap:.3rem;font-size:.68rem;color:#60a5fa;text-decoration:none;padding:2px 8px;border-radius:4px;border:1px solid #1e3a8a;background:#0c1a3d;transition:all .15s;}
.bfp-source-link:hover{background:#1e3a5f;color:#93c5fd;}
.bfp-conf-badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.68rem;padding:2px 8px;border-radius:999px;font-weight:600;}
.bfp-conf-high{background:#052e16;color:#4ade80;border:1px solid #166534;}
.bfp-conf-mid{background:#422006;color:#fbbf24;border:1px solid #92400e;}
.bfp-conf-low{background:#450a0a;color:#f87171;border:1px solid #991b1b;}
.bfp-dur-badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.68rem;padding:2px 8px;border-radius:999px;background:#111827;color:#94a3b8;border:1px solid #374151;}
.bfp-pipe-badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.68rem;padding:2px 8px;border-radius:999px;background:#0c1a3d;color:#818cf8;border:1px solid #312e81;}
.bfp-seg-badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.68rem;padding:2px 8px;border-radius:999px;background:#111827;color:#94a3b8;border:1px solid #374151;}
.bfp-mood-badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.68rem;padding:2px 8px;border-radius:999px;background:transparent;border:1px solid;}
.bfp-price-badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.68rem;padding:2px 8px;border-radius:999px;background:#052e16;color:#4ade80;border:1px solid #166534;font-weight:600;}
.bfp-mini-bar{display:inline-block;width:24px;height:6px;border-radius:3px;background:#1e293b;position:relative;overflow:hidden;vertical-align:middle;margin-right:3px;}
.bfp-mini-bar::after{content:'';position:absolute;left:0;top:0;height:100%;width:var(--pct,0%);background:var(--bar-color,#4ade80);border-radius:3px;transition:width .6s ease;}
.bfp-metrics-strip{margin-top:.5rem;padding:.5rem .6rem;background:#080c14;border:1px solid #1e293b;border-radius:6px;}
.bfp-strip-bar{height:4px;background:#1e293b;border-radius:2px;overflow:hidden;margin-bottom:.35rem;}
.bfp-strip-fill{height:100%;border-radius:2px;transition:width .8s ease;}
.bfp-strip-labels{display:flex;gap:.8rem;flex-wrap:wrap;font-size:.68rem;color:#64748b;}
`;
    document.head.appendChild(s);
  }

  /* ── Expose ───────────────────────────────────────────────── */
  global.BonfyrePanels = {
    init: injectStyles,
    renderPanels: renderPanels,
    renderQuality: renderQuality,
    renderTone: renderTone,
    renderRepurpose: renderRepurpose,
    renderWaterfall: renderWaterfall,
    renderOffer: renderOffer,
    renderSegmentHeatmap: renderSegmentHeatmap,
    renderLedger: renderLedger,
    renderTags: renderTags,
    mdToHtml: mdToHtml,
    load: load,
    loadText: loadText,
    autoEnhance: autoEnhance,
    _toggle: toggle
  };

})(typeof window !== 'undefined' ? window : this);
