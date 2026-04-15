// ==UserScript==
// @name         Ask Seattle Local Classifier Helper
// @namespace    https://github.com/local/ask-seattle
// @version      0.1.11
// @description  Adds auto-checking, skip, re-check, binary labeling, and transformer comparison cards for the local Ask Seattle classifier bridge.
// @match        https://www.reddit.com/r/*
// @match        https://new.reddit.com/r/*
// @match        https://old.reddit.com/r/*
// @match        https://www.reddit.com/r/*/comments/*
// @match        https://new.reddit.com/r/*/comments/*
// @match        https://old.reddit.com/r/*/comments/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

(function () {
  'use strict';

  const BRIDGE_URL = 'http://127.0.0.1:8765';
  const PANEL_ID = 'ask-seattle-local-helper';
  const QUEUE_KEY = 'askSeattlePostQueue';
  const AUTO_NEXT_KEY = 'askSeattleAutoNext';
  const HOTKEY_SKIP = 's';
  const HOTKEY_POSITIVE = 'p';
  const HOTKEY_NEGATIVE = 'n';
  const AUTO_CHECK_RETRIES = 8;
  const AUTO_CHECK_DELAY_MS = 400;
  const CHECK_TIMEOUT_MS = 10000;
  const COMPARISON_TIMEOUT_MS = 120000;
  const MODEL_DISPLAY_NAMES = {
    tfidf_recommended: 'TF-IDF',
    transformer_deberta_v3_small: 'DeBERTa-v3-small',
    transformer_modernbert_base: 'ModernBERT-base',
    transformer_neobert: 'NeoBERT',
    transformer_modernbert_large: 'ModernBERT-large',
  };
  let lastAutoCheckedKey = '';
  let currentCheckToken = 0;

  function textFrom(selector, root = document) {
    const node = root.querySelector(selector);
    return node ? (node.innerText || node.textContent || '').trim() : '';
  }

  function attrFrom(root, names) {
    if (!root || !root.getAttribute) return '';
    for (const name of names) {
      const value = root.getAttribute(name);
      if (value) return value.trim();
    }
    return '';
  }

  function postRoot() {
    return document.querySelector('shreddit-post') || document.querySelector('[data-testid="post-container"]') || document;
  }

  function parseCreatedUtc(value) {
    if (!value) return '';
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      return numeric > 10000000000 ? Math.round(numeric / 1000) : numeric;
    }
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? Math.round(parsed / 1000) : '';
  }

  function subredditFromUrl() {
    const match = window.location.pathname.match(/\/r\/([^/]+)/);
    return match ? match[1] : '';
  }

  function hostnameFromUrl(url) {
    try {
      return url ? new URL(url, window.location.origin).hostname : '';
    } catch (_error) {
      return '';
    }
  }

  function postMetadata(root) {
    const contentHref = attrFrom(root, ['content-href', 'url', 'href']);
    const postType = attrFrom(root, ['post-type', 'type']);
    const createdValue =
      attrFrom(root, ['created-timestamp', 'created-utc', 'created', 'created-at']) ||
      root.querySelector?.('time[datetime]')?.getAttribute('datetime') ||
      '';
    const rootText = (root.innerText || root.textContent || '').toLowerCase();
    const isCrosspost =
      postType.toLowerCase().includes('crosspost') ||
      Boolean(root.querySelector?.('[crosspost], [is-crosspost], crosspost-root')) ||
      rootText.includes('crossposted by');

    return {
      subreddit: attrFrom(root, ['subreddit-name', 'subreddit-prefixed-name']).replace(/^r\//, '') || subredditFromUrl(),
      created_utc: parseCreatedUtc(createdValue),
      post_type: postType,
      content_href: contentHref,
      content_domain: hostnameFromUrl(contentHref),
      is_crosspost: isCrosspost,
      capture_context: isCommentPage() ? 'comment_page' : 'listing_page',
    };
  }

  function currentPost() {
    const permalink = window.location.href.split('?')[0].split('#')[0];
    const idMatch = permalink.match(/\/comments\/([^/]+)/);
    const root = postRoot();
    const title =
      attrFrom(root, ['post-title']) ||
      textFrom('[slot="title"]', root) ||
      textFrom('h1', root) ||
      textFrom('[data-testid="post-container"] h1') ||
      textFrom('[data-test-id="post-content"] h1') ||
      (isCommentPage() ? textFrom('h1') : '') ||
      document.title.replace(/ : r\/.*$/, '').replace(/ - Reddit$/, '').trim();
    const selftext =
      textFrom('[slot="text-body"]', root) ||
      textFrom('[slot="body"]', root) ||
      textFrom('[data-click-id="text"]', root) ||
      textFrom('[data-test-id="post-content"] [data-click-id="text"]') ||
      '';

    return {
      id: idMatch ? idMatch[1] : '',
      permalink,
      title,
      selftext,
      collected_at: new Date().toISOString(),
      ...postMetadata(root),
    };
  }

  function postIdentity(post) {
    return post.id || normalizePostUrl(post.permalink) || '';
  }

  function isCommentPage() {
    return /\/comments\/[^/]+/.test(window.location.pathname);
  }

  function normalizePostUrl(url) {
    try {
      const parsed = new URL(url, window.location.origin);
      if (!/\/r\/[^/]+\/comments\/[^/]+/.test(parsed.pathname)) return '';
      return `${parsed.origin}${parsed.pathname.replace(/\/+$/, '')}/`;
    } catch (_error) {
      return '';
    }
  }

  function postIdFromUrl(url) {
    try {
      const parsed = new URL(url, window.location.origin);
      const idMatch = parsed.pathname.match(/\/comments\/([^/]+)/);
      return idMatch ? idMatch[1] : '';
    } catch (_error) {
      return '';
    }
  }

  function queueIndexForCurrentUrl(queue) {
    const currentUrl = normalizePostUrl(window.location.href);
    const currentId = postIdFromUrl(currentUrl);
    if (currentId) {
      const idIndex = queue.urls.findIndex((url) => postIdFromUrl(url) === currentId);
      if (idIndex >= 0) return idIndex;
    }
    return queue.urls.indexOf(currentUrl);
  }

  function collectVisiblePostQueue() {
    const seen = new Set();
    const urls = [];

    const candidates = [
      ...document.querySelectorAll('a[href*="/comments/"]'),
      ...document.querySelectorAll('shreddit-post'),
      ...document.querySelectorAll('[data-testid="post-container"]'),
      ...document.querySelectorAll('[data-fullname]'),
    ];

    for (const candidate of candidates) {
      const href =
        candidate.href ||
        candidate.getAttribute('href') ||
        candidate.getAttribute('permalink') ||
        candidate.getAttribute('content-href') ||
        candidate.querySelector?.('a[href*="/comments/"]')?.href ||
        '';
      const normalized = normalizePostUrl(href);
      if (!normalized || seen.has(normalized)) continue;
      seen.add(normalized);
      urls.push(normalized);
    }

    if (urls.length > 1) {
      localStorage.setItem(
        QUEUE_KEY,
        JSON.stringify({
          source: window.location.href,
          capturedAt: new Date().toISOString(),
          urls,
        })
      );
    }

    return urls;
  }

  function updateQueueStatus(urls = null) {
    const queueStatus = document.querySelector(`#${PANEL_ID} .ask-seattle-queue`);
    if (!queueStatus) return;

    const queue = urls ? { urls } : loadPostQueue();
    const index = queueIndexForCurrentUrl(queue);
    const position = index >= 0 ? `${index + 1}/${queue.urls.length}` : `0/${queue.urls.length}`;
    queueStatus.textContent = `Queue: ${position}`;
  }

  function seedQueueFromPage(showStatus = true) {
    const urls = collectVisiblePostQueue();
    updateQueueStatus(urls);
    if (showStatus) {
      if (urls.length > 1) {
        setStatus(`Seeded ${urls.length} visible posts.`);
      } else {
        setStatus(`Only found ${urls.length} post link. Scroll the listing and try again.`, true);
      }
    }
    return urls;
  }

  function loadPostQueue() {
    try {
      return JSON.parse(localStorage.getItem(QUEUE_KEY) || '{"urls":[]}');
    } catch (_error) {
      return { urls: [] };
    }
  }

  function isAutoNextEnabled() {
    return localStorage.getItem(AUTO_NEXT_KEY) === '1';
  }

  function setAutoNextEnabled(enabled) {
    localStorage.setItem(AUTO_NEXT_KEY, enabled ? '1' : '0');
  }

  function bridgePost(path, payload, options = {}) {
    const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : CHECK_TIMEOUT_MS;
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: 'POST',
        url: `${BRIDGE_URL}${path}`,
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify(payload),
        timeout: timeoutMs,
        onload: (response) => {
          try {
            const body = JSON.parse(response.responseText || '{}');
            if (response.status >= 200 && response.status < 300 && body.ok !== false) {
              resolve(body);
            } else {
              reject(new Error(body.error || `Bridge returned HTTP ${response.status}`));
            }
          } catch (error) {
            reject(error);
          }
        },
        onerror: () => reject(new Error('Could not reach the local Ask Seattle bridge.')),
        ontimeout: () => reject(new Error('Timed out waiting for the local Ask Seattle bridge.')),
      });
    });
  }

  function setStatus(message, isError = false) {
    const status = document.querySelector(`#${PANEL_ID} .ask-seattle-status`);
    if (!status) return;
    status.textContent = message;
    status.style.color = isError ? '#b00020' : '#1f6f43';
  }

  function setDecisionState(message, tone = 'neutral') {
    const verdict = document.querySelector(`#${PANEL_ID} .ask-seattle-verdict`);
    if (!verdict) return;

    verdict.textContent = message;
    verdict.style.borderColor = '#999';
    verdict.style.background = '#f2f2f2';
    verdict.style.color = '#333';

    if (tone === 'flag') {
      verdict.style.borderColor = '#b91c1c';
      verdict.style.background = '#fde8e8';
      verdict.style.color = '#8a1111';
    } else if (tone === 'pass') {
      verdict.style.borderColor = '#1f6f43';
      verdict.style.background = '#e8f5ec';
      verdict.style.color = '#1f6f43';
    } else if (tone === 'pending') {
      verdict.style.borderColor = '#8a6d1d';
      verdict.style.background = '#fff7e0';
      verdict.style.color = '#6f5717';
    } else if (tone === 'error') {
      verdict.style.borderColor = '#b00020';
      verdict.style.background = '#fdecef';
      verdict.style.color = '#b00020';
    }
  }

  function displayNameForModel(entry) {
    if (entry.display_name) return entry.display_name;
    const name = String(entry.name || entry.model_name || '');
    if (MODEL_DISPLAY_NAMES[name]) return MODEL_DISPLAY_NAMES[name];
    const raw = String(entry.name || entry.model_family || entry.model_name || '').toLowerCase();
    if (raw.includes('tfidf')) return 'TF-IDF';
    if (raw.includes('neobert')) return 'NeoBERT';
    if (raw.includes('modernbert-large')) return 'ModernBERT-large';
    if (raw.includes('modernbert')) return 'ModernBERT-base';
    if (raw.includes('deberta')) return 'DeBERTa-v3-small';
    return entry.name || entry.model_name || 'Model';
  }

  function transformerOnlyComparisons(entries) {
    return Array.isArray(entries) && entries.length > 0 && entries.every((entry) => String(entry.model_family || '').includes('transformer'));
  }

  function comparisonSectionTitle(entries) {
    const total = Array.isArray(entries) ? entries.length : 0;
    const label = transformerOnlyComparisons(entries) ? 'Transformer checks' : 'Comparison checks';
    if (!total) return label;
    return total > 4 ? `${label} (${total}, scroll)` : `${label} (${total})`;
  }

  function updateEvaluationLayout(entries) {
    const container = document.querySelector(`#${PANEL_ID} .ask-seattle-evaluations`);
    if (!container) return;
    const total = Array.isArray(entries) ? entries.length : 0;
    if (total <= 1) {
      container.style.gridTemplateColumns = '1fr';
      container.style.maxHeight = 'none';
      return;
    }
    if (total <= 4) {
      container.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
      container.style.maxHeight = 'none';
      return;
    }
    container.style.gridTemplateColumns = 'repeat(3, minmax(0, 1fr))';
    container.style.maxHeight = '50vh';
  }

  function resultTone(result) {
    if (result.label === 'askseattle' && result.confidence_band === 'high') return 'flag';
    if (result.label === 'askseattle') return 'pending';
    return 'pass';
  }

  function setEvaluationResultsPending(message) {
    const container = document.querySelector(`#${PANEL_ID} .ask-seattle-evaluations`);
    if (!container) return;
    updateEvaluationTitle([]);
    updateEvaluationLayout([]);

    const row = document.createElement('div');
    row.textContent = message;
    row.style.border = '1px solid #d0d0d0';
    row.style.borderRadius = '6px';
    row.style.padding = '6px 8px';
    row.style.background = '#fff7e0';
    row.style.color = '#6f5717';
    container.replaceChildren(row);
  }

  function loadingComparisonEntries(models) {
    return (models || []).map((model) => ({
      ...model,
      loading: true,
    }));
  }

  function updateEvaluationTitle(entries) {
    const title = document.querySelector(`#${PANEL_ID} .ask-seattle-evaluations-title`);
    if (!title) return;
    title.textContent = comparisonSectionTitle(entries);
  }

  function comparisonStatusText(baseStatusText, completed, models) {
    const total = Array.isArray(models) ? models.length : 0;
    if (!total || completed >= total) return baseStatusText;
    const label = transformerOnlyComparisons(models) ? 'transformer checks' : 'comparison checks';
    return `${baseStatusText} | ${label} ${completed}/${total}`;
  }

  function renderEvaluationResults(entries) {
    const container = document.querySelector(`#${PANEL_ID} .ask-seattle-evaluations`);
    if (!container) return;
    updateEvaluationTitle(entries);
    updateEvaluationLayout(entries);
    container.replaceChildren();

    if (!entries || entries.length === 0) {
      const row = document.createElement('div');
      row.textContent = 'No supported comparison models loaded.';
      row.style.border = '1px solid #d0d0d0';
      row.style.borderRadius = '6px';
      row.style.padding = '6px 8px';
      row.style.background = '#f2f2f2';
      row.style.color = '#555';
      row.style.gridColumn = '1 / -1';
      container.append(row);
      return;
    }

    for (const entry of entries) {
      if (entry.loading) {
        const pendingRow = document.createElement('div');
        pendingRow.style.display = 'flex';
        pendingRow.style.flexDirection = 'column';
        pendingRow.style.gap = '6px';
        pendingRow.style.border = '1px solid #8a6d1d';
        pendingRow.style.borderRadius = '6px';
        pendingRow.style.padding = '8px';
        pendingRow.style.background = '#fff7e0';
        pendingRow.style.minHeight = '88px';

        const modelLabel = document.createElement('div');
        modelLabel.textContent = displayNameForModel(entry);
        modelLabel.style.fontWeight = '600';
        modelLabel.style.fontSize = '12px';

        const verdict = document.createElement('div');
        verdict.textContent = 'LOADING...';
        verdict.style.fontWeight = '700';
        verdict.style.fontSize = '13px';
        verdict.style.lineHeight = '1.2';
        verdict.style.color = '#6f5717';

        const detail = document.createElement('div');
        detail.textContent = 'Waiting for model result';
        detail.style.fontSize = '11px';
        detail.style.lineHeight = '1.3';
        detail.style.color = '#6f5717';

        pendingRow.append(modelLabel, verdict, detail);
        container.append(pendingRow);
        continue;
      }

      if (entry.error) {
        const errorRow = document.createElement('div');
        errorRow.style.display = 'flex';
        errorRow.style.flexDirection = 'column';
        errorRow.style.gap = '6px';
        errorRow.style.border = '1px solid #b00020';
        errorRow.style.borderRadius = '6px';
        errorRow.style.padding = '8px';
        errorRow.style.background = '#fdecef';
        errorRow.style.minHeight = '88px';

        const modelLabel = document.createElement('div');
        modelLabel.textContent = displayNameForModel(entry);
        modelLabel.style.fontWeight = '600';
        modelLabel.style.fontSize = '12px';

        const verdict = document.createElement('div');
        verdict.textContent = 'CHECK FAILED';
        verdict.style.fontWeight = '700';
        verdict.style.fontSize = '13px';
        verdict.style.lineHeight = '1.2';
        verdict.style.color = '#b00020';

        const detail = document.createElement('div');
        detail.textContent = String(entry.error || 'Unknown comparison error');
        detail.style.fontSize = '11px';
        detail.style.lineHeight = '1.3';
        detail.style.color = '#8a1111';
        detail.style.wordBreak = 'break-word';

        errorRow.append(modelLabel, verdict, detail);
        container.append(errorRow);
        continue;
      }

      const result = entry.result || {};
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.flexDirection = 'column';
      row.style.gap = '6px';
      row.style.border = '1px solid #d0d0d0';
      row.style.borderRadius = '6px';
      row.style.padding = '8px';
      row.style.background = '#fff';
      row.style.minHeight = '88px';

      const tone = resultTone(result);
      if (tone === 'flag') {
        row.style.borderColor = '#b91c1c';
        row.style.background = '#fde8e8';
      } else if (tone === 'pending') {
        row.style.borderColor = '#8a6d1d';
        row.style.background = '#fff7e0';
      } else {
        row.style.borderColor = '#1f6f43';
        row.style.background = '#e8f5ec';
      }

      const header = document.createElement('div');
      header.style.display = 'flex';
      header.style.alignItems = 'center';
      header.style.justifyContent = 'space-between';
      header.style.gap = '6px';

      const modelLabel = document.createElement('div');
      modelLabel.textContent = displayNameForModel(entry);
      modelLabel.style.fontWeight = '600';
      modelLabel.style.fontSize = '12px';
      modelLabel.style.letterSpacing = '0';

      const score = document.createElement('div');
      const value = Number(result.score);
      score.textContent = Number.isFinite(value) ? value.toFixed(3) : '-';
      score.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, monospace';
      score.style.fontSize = '12px';
      score.style.opacity = '0.85';

      header.append(modelLabel, score);

      const verdict = document.createElement('div');
      verdict.style.fontWeight = '700';
      verdict.style.fontSize = '13px';
      verdict.style.lineHeight = '1.2';
      verdict.textContent = result.label === 'askseattle' ? 'ASKSEATTLE' : 'NOT ASKSEATTLE';

      const outcome = document.createElement('div');
      const band = String(result.confidence_band || 'unknown').toUpperCase();
      outcome.textContent = band === 'HIGH' ? 'high confidence' : band === 'BORDERLINE' ? 'borderline' : 'low confidence';
      outcome.style.fontSize = '12px';
      outcome.style.opacity = '0.85';

      row.append(header, verdict, outcome);
      container.append(row);
    }
  }

  async function loadComparisonResults(checkToken, post, models, baseStatusText) {
    if (!models || models.length === 0) {
      renderEvaluationResults([]);
      setStatus(baseStatusText);
      return;
    }

    const entries = loadingComparisonEntries(models);
    let completed = 0;
    renderEvaluationResults(entries);
    setStatus(comparisonStatusText(baseStatusText, completed, models));

    await Promise.all(
      models.map(async (model, index) => {
        try {
          const response = await bridgePost(
            '/check-comparison',
            {
              ...post,
              name: model.name,
            },
            { timeoutMs: COMPARISON_TIMEOUT_MS }
          );
          if (checkToken !== currentCheckToken) return;
          entries[index] = response.comparison || { ...model, error: 'Missing comparison payload' };
        } catch (error) {
          if (checkToken !== currentCheckToken) return;
          entries[index] = {
            ...model,
            error: error.message || 'Unknown comparison error',
          };
        }
        if (checkToken !== currentCheckToken) return;
        completed += 1;
        renderEvaluationResults(entries);
        setStatus(comparisonStatusText(baseStatusText, completed, models));
      })
    );
  }

  function autoRetrainStatusText(autoRetrain) {
    if (!autoRetrain || autoRetrain.enabled !== true) return '';
    if (autoRetrain.scheduled) {
      return ` Auto-retrain started at ${autoRetrain.training_records} prepared rows.`;
    }
    if (autoRetrain.in_progress) {
      return ' Auto-retrain already running.';
    }
    if (typeof autoRetrain.labels_until_retrain === 'number') {
      return ` ${autoRetrain.labels_until_retrain} prepared row(s) until auto-retrain.`;
    }
    return '';
  }

  function button(label, onClick) {
    const element = document.createElement('button');
    element.type = 'button';
    element.textContent = label;
    element.style.border = '1px solid #888';
    element.style.borderRadius = '6px';
    element.style.background = '#fff';
    element.style.color = '#111';
    element.style.padding = '6px 9px';
    element.style.cursor = 'pointer';
    element.addEventListener('click', onClick);
    return element;
  }

  async function checkPost(options = {}) {
    const { post: providedPost = null, auto = false } = options;
    const checkToken = ++currentCheckToken;
    if (!isCommentPage()) {
      setStatus('Open a post page before checking or training.', true);
      return;
    }
    const post = providedPost || currentPost();
    if (!post.title) {
      if (!auto) {
        setStatus('Could not find a post title on this page.', true);
      }
      return;
    }

    setStatus(auto ? 'Auto-checking...' : 'Checking...');
    setDecisionState('Checking current post...', 'pending');
    setEvaluationResultsPending('Checking transformer cards...');
    try {
      const response = await bridgePost('/check', { ...post, include_comparisons: false });
      if (checkToken !== currentCheckToken) return;
      const result = response.result;
      const bandLabel = String(result.confidence_band || '').toUpperCase();
      const verdictMessage =
        result.label === 'askseattle'
          ? bandLabel === 'HIGH'
            ? 'Looks like askseattle (high confidence)'
            : 'Looks like askseattle (borderline)'
          : 'Does not look like askseattle';
      setDecisionState(verdictMessage, result.label === 'askseattle' ? 'flag' : 'pass');
      const baseStatusText = `${result.confidence_band} ${result.label} score=${result.score.toFixed(3)} low=${result.low_threshold} high=${result.high_threshold}`;
      const comparisonModels = Array.isArray(response.comparison_models) ? response.comparison_models : [];
      void loadComparisonResults(checkToken, post, comparisonModels, baseStatusText);
    } catch (error) {
      if (checkToken !== currentCheckToken) return;
      setDecisionState('Check failed', 'error');
      setEvaluationResultsPending('Check failed.');
      setStatus(error.message, true);
    }
  }

  function scheduleAutoCheck(attempt = 0) {
    if (!isCommentPage()) return;
    const post = currentPost();
    const identity = postIdentity(post);
    if (!identity || identity === lastAutoCheckedKey) return;

    window.setTimeout(() => {
      if (!isCommentPage()) return;
      const freshPost = currentPost();
      const freshIdentity = postIdentity(freshPost);
      if (!freshIdentity || freshIdentity !== identity) return;
      if (!freshPost.title) {
        if (attempt + 1 < AUTO_CHECK_RETRIES) {
          scheduleAutoCheck(attempt + 1);
        }
        return;
      }

      lastAutoCheckedKey = freshIdentity;
      checkPost({ post: freshPost, auto: true });
    }, AUTO_CHECK_DELAY_MS);
  }

  async function refreshRecordedStatus() {
    if (!isCommentPage()) return;
    const post = currentPost();
    const recorded = document.querySelector(`#${PANEL_ID} .ask-seattle-recorded`);
    if (!recorded || (!post.id && !post.permalink)) return;

    try {
      const response = await bridgePost('/recorded', post);
      if (response.recorded) {
        recorded.textContent = `Recorded: ${response.record.label}`;
        recorded.style.color = '#1f6f43';
      } else {
        recorded.textContent = 'Not recorded yet';
        recorded.style.color = '#555';
      }
    } catch (error) {
      recorded.textContent = `Recorded check failed: ${error.message}`;
      recorded.style.color = '#b00020';
    }
  }

  async function trainPost(label) {
    if (!isCommentPage()) {
      setStatus('Open a post page before checking or training.', true);
      return;
    }
    const post = currentPost();
    if (!post.title) {
      setStatus('Could not find a post title on this page.', true);
      return;
    }

    setStatus(`Saving ${label} label...`);
    try {
      const response = await bridgePost('/train', { ...post, label });
      setStatus(
        `Saved ${response.saved.label} to ${response.label_path}.${autoRetrainStatusText(response.auto_retrain)}`
      );
      await refreshRecordedStatus();
      if (isAutoNextEnabled()) {
        setTimeout(moveToNextPost, 500);
      }
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  function moveToNextPost() {
    const queue = loadPostQueue();
    const index = queueIndexForCurrentUrl(queue);
    if (index >= 0 && index + 1 < queue.urls.length) {
      window.location.href = queue.urls[index + 1];
      return;
    }

    setStatus('No next queued post found. Open the subreddit listing first to seed the queue.', true);
  }

  function isTypingTarget(target) {
    if (!target) return false;
    const tagName = String(target.tagName || '').toLowerCase();
    return (
      tagName === 'input' ||
      tagName === 'textarea' ||
      tagName === 'select' ||
      Boolean(target.isContentEditable)
    );
  }

  function handleHotkeys(event) {
    if (!isCommentPage() || isTypingTarget(event.target) || event.repeat) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const key = String(event.key || '').toLowerCase();
    if (key === HOTKEY_SKIP) {
      event.preventDefault();
      moveToNextPost();
    } else if (key === HOTKEY_POSITIVE) {
      event.preventDefault();
      trainPost('askseattle');
    } else if (key === HOTKEY_NEGATIVE) {
      event.preventDefault();
      trainPost('not_askseattle');
    }
  }

  function mountPanel() {
    if (document.getElementById(PANEL_ID)) return;

    const panel = document.createElement('div');
    panel.id = PANEL_ID;
    panel.style.position = 'fixed';
    panel.style.top = '96px';
    panel.style.right = '16px';
    panel.style.zIndex = '999999';
    panel.style.display = 'flex';
    panel.style.flexDirection = 'column';
    panel.style.gap = '8px';
    panel.style.width = '380px';
    panel.style.padding = '10px';
    panel.style.border = '1px solid #777';
    panel.style.borderRadius = '8px';
    panel.style.background = '#f7f7f7';
    panel.style.color = '#111';
    panel.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.2)';
    panel.style.fontFamily = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    panel.style.fontSize = '13px';
    panel.style.maxHeight = 'calc(100vh - 112px)';
    panel.style.overflow = 'hidden';

    const title = document.createElement('strong');
    title.textContent = 'Ask Seattle';

    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.flexWrap = 'wrap';
    row.style.gap = '8px';
    row.append(
      button('Seed queue', () => seedQueueFromPage(true)),
      button('Skip (S)', () => moveToNextPost()),
      button('Re-check', () => checkPost()),
      button('Train positive (P)', () => trainPost('askseattle')),
      button('Train negative (N)', () => trainPost('not_askseattle'))
    );

    const autoNextLabel = document.createElement('label');
    autoNextLabel.style.display = 'flex';
    autoNextLabel.style.alignItems = 'center';
    autoNextLabel.style.gap = '6px';

    const autoNext = document.createElement('input');
    autoNext.type = 'checkbox';
    autoNext.checked = isAutoNextEnabled();
    autoNext.addEventListener('change', () => setAutoNextEnabled(autoNext.checked));
    autoNextLabel.append(autoNext, document.createTextNode('Auto next after training'));

    const recorded = document.createElement('div');
    recorded.className = 'ask-seattle-recorded';
    recorded.textContent = 'Checking recorded status...';
    recorded.style.lineHeight = '1.35';
    recorded.style.wordBreak = 'break-word';

    const verdict = document.createElement('div');
    verdict.className = 'ask-seattle-verdict';
    verdict.textContent = isCommentPage() ? 'Waiting for auto-check...' : 'Open a post to check it';
    verdict.style.lineHeight = '1.35';
    verdict.style.wordBreak = 'break-word';
    verdict.style.padding = '8px';
    verdict.style.border = '1px solid #999';
    verdict.style.borderRadius = '6px';
    verdict.style.background = '#f2f2f2';
    verdict.style.color = '#333';
    verdict.style.fontWeight = '600';

    const evaluationsTitle = document.createElement('div');
    evaluationsTitle.className = 'ask-seattle-evaluations-title';
    evaluationsTitle.textContent = 'Transformer checks';
    evaluationsTitle.style.fontWeight = '600';

    const evaluations = document.createElement('div');
    evaluations.className = 'ask-seattle-evaluations';
    evaluations.style.display = 'grid';
    evaluations.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
    evaluations.style.gap = '6px';
    evaluations.style.maxHeight = 'none';
    evaluations.style.overflowY = 'auto';
    evaluations.style.paddingRight = '2px';
    evaluations.style.alignContent = 'start';

    const queueStatus = document.createElement('div');
    queueStatus.className = 'ask-seattle-queue';
    queueStatus.textContent = 'Queue: checking...';
    queueStatus.style.lineHeight = '1.35';
    queueStatus.style.wordBreak = 'break-word';

    const status = document.createElement('div');
    status.className = 'ask-seattle-status';
    status.textContent = 'Bridge: 127.0.0.1:8765';
    status.style.lineHeight = '1.35';
    status.style.wordBreak = 'break-word';

    panel.append(title, row, autoNextLabel, queueStatus, recorded, verdict, evaluationsTitle, evaluations, status);
    document.body.append(panel);
    setEvaluationResultsPending(isCommentPage() ? 'Waiting for auto-check...' : 'Open a post to check it');
    updateQueueStatus();
    refreshRecordedStatus();
    scheduleAutoCheck();
  }

  function watchSpaNavigation() {
    let lastUrl = window.location.href;
    window.setInterval(() => {
      if (window.location.href === lastUrl) return;
      currentCheckToken += 1;
      lastUrl = window.location.href;
      if (!isCommentPage()) {
        seedQueueFromPage(false);
        setDecisionState('Open a post to check it');
        setEvaluationResultsPending('Open a post to check it');
      }
      if (isCommentPage()) {
        mountPanel();
        setDecisionState('Waiting for auto-check...', 'pending');
        setEvaluationResultsPending('Waiting for auto-check...');
        updateQueueStatus();
        refreshRecordedStatus();
        scheduleAutoCheck();
      }
    }, 1000);
  }

  document.addEventListener('mousedown', (event) => {
    const anchor = event.target.closest?.('a[href*="/comments/"]');
    if (anchor && !isCommentPage()) {
      seedQueueFromPage(false);
    }
  }, true);

  mountPanel();
  if (!isCommentPage()) {
    seedQueueFromPage(false);
    setDecisionState('Open a post to check it');
    setEvaluationResultsPending('Open a post to check it');
  } else {
    setDecisionState('Waiting for auto-check...', 'pending');
    setEvaluationResultsPending('Waiting for auto-check...');
    scheduleAutoCheck();
  }
  document.addEventListener('keydown', handleHotkeys, true);
  watchSpaNavigation();
})();
