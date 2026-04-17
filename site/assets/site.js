(function () {
  var path = window.location.pathname;
  var base = path.indexOf("/bonfyre-oss/") === 0 || path === "/bonfyre-oss" ? "/bonfyre-oss" : "";
  var APP_LINKS = {
    "Shift Handoff Board": "https://nickgonzales76017.github.io/pages-shift-handoff/",
    "Memory Atlas": "https://nickgonzales76017.github.io/pages-memory-atlas/",
    "Freelancer Evidence Vault": "https://nickgonzales76017.github.io/pages-freelancer-evidence/",
    "Customer Voice Board": "https://nickgonzales76017.github.io/pages-customer-voice/",
    "Family History Museum": "https://nickgonzales76017.github.io/pages-family-history/",
    "Podcast Plant": "https://nickgonzales76017.github.io/pages-podcast-plant/",
    "Postmortem Atlas": "https://nickgonzales76017.github.io/pages-postmortem-atlas/",
    "Explain This Repo": "https://nickgonzales76017.github.io/pages-explain-repo/",
    "Town Box": "https://nickgonzales76017.github.io/pages-town-box/",
    "Grant Evidence Pack": "https://nickgonzales76017.github.io/pages-grant-evidence/",
    "Micro-Consulting Storefront": "https://nickgonzales76017.github.io/pages-micro-consulting/",
    "Personal Legal Prep Binder": "https://nickgonzales76017.github.io/pages-legal-prep/",
    "OSS Maintainer Cockpit": "https://nickgonzales76017.github.io/pages-oss-cockpit/",
    "Release-Note Radio": "https://nickgonzales76017.github.io/pages-release-radio/",
    "Async Standup Newspaper": "https://nickgonzales76017.github.io/pages-async-standup/",
    "Competitive Intelligence Scrapbook": "https://nickgonzales76017.github.io/pages-competitive-intel/",
    "Sales Call Distiller": "https://nickgonzales76017.github.io/pages-sales-distiller/",
    "Procurement Memory Site": "https://nickgonzales76017.github.io/pages-procurement-memory/",
    "Museum Exhibit Builder": "https://nickgonzales76017.github.io/pages-museum-exhibit/",
    "Local Archive Explorer": "https://nickgonzales76017.github.io/pages-local-archive/"
  };

  function siteUrl(relativePath) {
    if (!relativePath) {
      return base || "";
    }

    if (relativePath.indexOf("http") === 0) {
      return relativePath;
    }

    return base + relativePath;
  }

  function fetchJson(relativePath) {
    return window.fetch(siteUrl(relativePath)).then(function (response) {
      if (!response.ok) {
        throw new Error("Request failed: " + relativePath);
      }
      return response.json();
    });
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function updateStatus(status) {
    document.querySelectorAll("[data-status-key]").forEach(function (node) {
      var key = node.getAttribute("data-status-key");
      if (Object.prototype.hasOwnProperty.call(status, key)) {
        node.textContent = status[key];
      }
    });
  }

  function renderAppsPage(appData, recipeData) {
    var categoryNode = document.getElementById("category-summary");
    var appsNode = document.getElementById("apps-grid");
    var recipesNode = document.getElementById("recipes-grid");

    if (categoryNode) {
      var counts = {};
      appData.apps.forEach(function (app) {
        counts[app.category] = (counts[app.category] || 0) + 1;
      });
      categoryNode.innerHTML = Object.keys(counts)
        .sort()
        .map(function (category) {
          return '<span class="pill">' + escapeHtml(category) + " x " + counts[category] + "</span>";
        })
        .join("");
    }

    if (appsNode) {
      appsNode.innerHTML = appData.apps
        .map(function (app) {
          var target = APP_LINKS[app.name] || "#";
          var external = target.indexOf("http") === 0 ? ' target="_blank" rel="noreferrer"' : "";
          return (
            '<a class="app-card" href="' + escapeHtml(target) + '"' + external + ">" +
            '<span class="card-meta">' + escapeHtml(app.category) + "</span>" +
            "<h3>" + escapeHtml(app.name) + "</h3>" +
            "<p>" + escapeHtml(app.description) + "</p>" +
            '<div class="pill-row"><span class="pill">Live page</span><span class="pill">Bonfyre Pages</span></div>' +
            "</a>"
          );
        })
        .join("");
    }

    if (recipesNode) {
      recipesNode.innerHTML = recipeData.recipes
        .map(function (recipe) {
          return (
            '<article class="recipe-card">' +
            '<span class="card-meta">' + escapeHtml(recipe.industry) + "</span>" +
            "<h3>" + escapeHtml(recipe.title) + "</h3>" +
            "<p><strong>Input:</strong> " + escapeHtml(recipe.input_desc) + "</p>" +
            "<p><strong>Output:</strong> " + escapeHtml(recipe.output_desc) + "</p>" +
            "<p><strong>Result:</strong> " + escapeHtml(recipe.result) + "</p>" +
            "</article>"
          );
        })
        .join("");
    }
  }

  function renderDocsPage(modelData, benchmarkData, hardwareData, binaryData, moqData) {
    var modelsNode = document.getElementById("fpq-models");
    var benchmarksNode = document.getElementById("benchmark-grid");
    var hardwareNode = document.getElementById("hardware-grid");
    var binariesNode = document.getElementById("binary-groups");
    var moqNode = document.getElementById("moq-features");

    if (modelsNode) {
      var rows = modelData.models
        .slice(0, 8)
        .map(function (model) {
          var download = model.hf_link
            ? '<a href="' + escapeHtml(model.hf_link) + '" target="_blank" rel="noreferrer">Model</a>'
            : "Internal run";
          return (
            "<tr>" +
            "<td><strong>" + escapeHtml(model.name) + "</strong><br>" + escapeHtml(model.domain) + "</td>" +
            "<td>" + escapeHtml(model.original_size) + "</td>" +
            "<td>" + escapeHtml(model.compressed_size) + "</td>" +
            "<td>" + escapeHtml(model.avg_cos) + "</td>" +
            "<td>" + escapeHtml(model.avg_bpw) + "</td>" +
            "<td>" + download + "</td>" +
            "</tr>"
          );
        })
        .join("");
      modelsNode.innerHTML =
        '<div class="table-card"><div class="table-wrap"><table class="data-table"><thead><tr><th>Model</th><th>Original</th><th>Compressed</th><th>Avg Cos</th><th>Avg BPW</th><th>Link</th></tr></thead><tbody>' +
        rows +
        "</tbody></table></div></div>";
    }

    if (benchmarksNode) {
      benchmarksNode.innerHTML = benchmarkData.benchmarks
        .slice(0, 8)
        .map(function (item) {
          return (
            '<article class="feature-card">' +
            '<span class="card-meta">' + escapeHtml(item.improvement) + "</span>" +
            "<h3>" + escapeHtml(item.metric) + "</h3>" +
            '<div class="detail-list">' +
            '<div class="detail-item"><strong>Before:</strong> ' + escapeHtml(item.before) + "</div>" +
            '<div class="detail-item"><strong>After:</strong> ' + escapeHtml(item.after) + "</div>" +
            "</div></article>"
          );
        })
        .join("");
    }

    if (hardwareNode) {
      hardwareNode.innerHTML = hardwareData.configs
        .map(function (config) {
          return (
            '<article class="compare-card">' +
            '<span class="card-meta">' + escapeHtml(config.cost) + "</span>" +
            "<h3>" + escapeHtml(config.device) + "</h3>" +
            "<p><strong>RAM:</strong> " + escapeHtml(config.ram) + "</p>" +
            '<div class="detail-list">' +
            '<div class="detail-item"><strong>Before FPQ:</strong> ' + escapeHtml(config.before_fpq) + "</div>" +
            '<div class="detail-item"><strong>After FPQ:</strong> ' + escapeHtml(config.after_fpq) + "</div>" +
            "</div></article>"
          );
        })
        .join("");
    }

    if (binariesNode) {
      var groupOrder = ["substrate", "transform", "surface", "value", "libraries"];
      binariesNode.innerHTML = groupOrder
        .map(function (group) {
          var items = binaryData.binaries[group] || [];
          return (
            '<section class="binary-group">' +
            '<span class="card-meta">' + escapeHtml(group) + " layer</span>" +
            "<h3>" + escapeHtml(group.charAt(0).toUpperCase() + group.slice(1)) + "</h3>" +
            "<p>" + items.length + " binaries in this layer.</p>" +
            '<div class="binary-items">' +
            items
              .map(function (item) {
                return (
                  '<div class="binary-item">' +
                  '<span class="binary-name">' + escapeHtml(item.name) + "</span>" +
                  '<span class="binary-size">' + escapeHtml(item.size) + "</span>" +
                  '<span class="binary-desc">' + escapeHtml(item.description) + "</span>" +
                  "</div>"
                );
              })
              .join("") +
            "</div></section>"
          );
        })
        .join("");
    }

    if (moqNode) {
      moqNode.innerHTML = moqData.features
        .map(function (feature) {
          return '<span class="pill">' + escapeHtml(feature) + "</span>";
        })
        .join("");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    fetchJson("/api/status")
      .then(updateStatus)
      .catch(function () {});

    if (document.getElementById("apps-grid") || document.getElementById("recipes-grid")) {
      Promise.all([fetchJson("/api/apps"), fetchJson("/api/recipes")])
        .then(function (payload) {
          renderAppsPage(payload[0], payload[1]);
        })
        .catch(function () {});
    }

    if (document.getElementById("fpq-models") || document.getElementById("binary-groups")) {
      Promise.all([
        fetchJson("/api/fpq/models"),
        fetchJson("/api/benchmarks"),
        fetchJson("/api/hardware"),
        fetchJson("/api/binaries"),
        fetchJson("/api/moq")
      ])
        .then(function (payload) {
          renderDocsPage(payload[0], payload[1], payload[2], payload[3], payload[4]);
        })
        .catch(function () {});
    }
  });
})();
