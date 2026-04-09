#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2) + '\n');
}

function slugify(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'record';
}

function pickArray(value, fallback) {
  return Array.isArray(value) && value.length ? value : fallback;
}

function buildRecord(app, source, index) {
  const tags = pickArray(source.tags, app.default_tags || []);
  const outputs = pickArray(source.outputs, app.default_outputs || []);
  return {
    id: source.id || `${slugify(app.slug || app.repo)}-${String(index + 1).padStart(3, '0')}`,
    file: source.file || source.title || `${app.title} source ${index + 1}`,
    time: source.time || source.published_at || 'Reference corpus',
    status: source.status || 'complete',
    brief: source.brief || app.default_brief || `${app.title} processed this public source into reusable outputs.`,
    tags,
    flagged: Boolean(source.flagged),
    demo: true,
    outputs,
    outputNotes: source.outputNotes || {},
    outputLinks: source.outputLinks || {},
    sourceLinks: pickArray(source.sourceLinks, []),
    sourceTitle: source.sourceTitle || source.title || '',
    sourceUrl: source.sourceUrl || source.public_url || '',
    sourceLabel: source.sourceLabel || 'Watch original public source',
    sourceCopy: source.sourceCopy || app.source_copy || 'Trace this card back to the public origin Bonfyre processed.',
    publisher: source.publisher || '',
    license: source.license || '',
    searchSummary: source.searchSummary || tags.slice(0, 3).join(', '),
    whyItMatters: source.whyItMatters || app.why_it_matters || `${app.title} becomes convincing when each record is tied back to a real public origin.`,
    searchIntro: source.searchIntro || app.search_intro || `Search across the ${app.slug} corpus to see how Bonfyre handles many real public sources.`,
    searchOutputs: pickArray(source.searchOutputs, outputs.slice(0, 1))
  };
}

function main() {
  const manifestPath = process.argv[2];
  if (!manifestPath) {
    console.error('usage: generate_reference_corpora_from_sources.mjs <manifest.json>');
    process.exit(1);
  }

  const manifest = readJson(manifestPath);
  for (const app of manifest.apps || []) {
    if (!Array.isArray(app.sources) || !app.sources.length) {
      throw new Error(`App ${app.repo} has no reviewed sources`);
    }
    const items = app.sources.map((source, index) => buildRecord(app, source, index));
    const outPath = path.resolve(app.out_path);
    writeJson(outPath, items);
    console.log(`${app.repo}: wrote ${items.length} provenance-backed records -> ${outPath}`);
  }
}

main();
