/**
 * Results Dashboard Logic (tts_app)
 * Fetches unblinded ratings and renders summary metric cards and Chart.js charts.
 */

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const res = await fetch("/api/results");
    const data = await res.json();

    if (data.status !== "success" || data.total_ratings === 0) {
      document.getElementById("stat-total-ratings").textContent = "0";
      document.getElementById("stat-evaluators-count").textContent = "No evaluators yet";
      document.getElementById("stat-baseline-mos").textContent = "N/A";
      document.getElementById("stat-clustered-mos").textContent = "N/A";
      document.getElementById("stat-pref-win").textContent = "N/A";
      document.getElementById("stat-pref-desc").textContent = "No data available";
      document.getElementById("lang-table-body").innerHTML = `
        <tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 32px;">
          No evaluations completed yet. <a href="/" style="color: var(--accent-primary); font-weight: 700;">Start the first session!</a>
        </td></tr>
      `;
      document.getElementById("evaluators-chips").innerHTML = `<span style="color: var(--text-muted);">None yet</span>`;
      return;
    }

    // 1. Fill Key Metric Cards
    document.getElementById("stat-total-ratings").textContent = data.total_ratings;
    document.getElementById("stat-evaluators-count").textContent = `${data.evaluators_count} evaluator${data.evaluators_count > 1 ? "s" : ""}`;

    const bOverall = data.overall.baseline;
    const cOverall = data.overall.clustered;

    document.getElementById("stat-baseline-mos").textContent = bOverall.mean.toFixed(2);
    document.getElementById("stat-baseline-ci").textContent = `95% CI: [${bOverall.ci_lo.toFixed(2)}, ${bOverall.ci_hi.toFixed(2)}]`;

    document.getElementById("stat-clustered-mos").textContent = cOverall.mean.toFixed(2);
    document.getElementById("stat-clustered-ci").textContent = `95% CI: [${cOverall.ci_lo.toFixed(2)}, ${cOverall.ci_hi.toFixed(2)}]`;

    const prefs = data.preferences;
    const totalPref = prefs.baseline + prefs.clustered + prefs.none;
    const pctBase = totalPref > 0 ? Math.round((prefs.baseline / totalPref) * 100) : 0;
    const pctClust = totalPref > 0 ? Math.round((prefs.clustered / totalPref) * 100) : 0;
    const pctNone = totalPref > 0 ? Math.round((prefs.none / totalPref) * 100) : 0;

    let winText = "Equivalent";
    if (prefs.baseline > prefs.clustered) winText = "Baseline Favored";
    else if (prefs.clustered > prefs.baseline) winText = "Clustered Favored";

    document.getElementById("stat-pref-win").textContent = winText;
    document.getElementById("stat-pref-desc").textContent = `Base: ${pctBase}% • Clust: ${pctClust}% • Equal: ${pctNone}%`;

    // 2. Render Evaluators Chips
    const chipsContainer = document.getElementById("evaluators-chips");
    chipsContainer.innerHTML = data.evaluators.map(name => `
      <span class="chip active" style="font-size: 0.85rem; padding: 6px 14px;">
        👤 ${name}
      </span>
    `).join("");

    // 3. Render Table
    const tableBody = document.getElementById("lang-table-body");
    const langCodes = Object.keys(data.by_language);

    tableBody.innerHTML = langCodes.map(code => {
      const item = data.by_language[code];
      const b = item.baseline;
      const c = item.clustered;
      const diff = (b.mean - c.mean).toFixed(3);
      const diffStr = (b.mean >= c.mean) ? `+${diff}` : `${diff}`;

      let prefWinner = "Equal";
      if (item.preferences.baseline > item.preferences.clustered) prefWinner = "Baseline";
      else if (item.preferences.clustered > item.preferences.baseline) prefWinner = "Clustered";

      return `
        <tr>
          <td style="font-weight: 700;">${item.name}</td>
          <td>${b.n}</td>
          <td style="color: #818cf8; font-weight: 600;">${b.mean.toFixed(3)} ± ${b.std.toFixed(2)}</td>
          <td style="color: #fb923c; font-weight: 600;">${c.mean.toFixed(3)} ± ${c.std.toFixed(2)}</td>
          <td style="font-weight: 700;">${diffStr}</td>
          <td><span class="brand-badge" style="background: var(--bg-elevated); color: var(--text-primary);">${prefWinner}</span></td>
        </tr>
      `;
    }).join("");

    // 4. Chart.js Defaults
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "'Inter', sans-serif";

    // 5. MOS Bar Chart
    const ctxBar = document.getElementById("mosBarChart").getContext("2d");
    const barLabels = langCodes.map(c => data.by_language[c].name);
    barLabels.push("Overall Pooled");

    const baselineMeans = langCodes.map(c => data.by_language[c].baseline.mean);
    baselineMeans.push(bOverall.mean);

    const clusteredMeans = langCodes.map(c => data.by_language[c].clustered.mean);
    clusteredMeans.push(cOverall.mean);

    new Chart(ctxBar, {
      type: "bar",
      data: {
        labels: barLabels,
        datasets: [
          {
            label: "Baseline (57 phonemes)",
            data: baselineMeans,
            backgroundColor: "rgba(99, 102, 241, 0.8)",
            borderColor: "#6366f1",
            borderWidth: 1,
            borderRadius: 6,
          },
          {
            label: "Clustered (39 clusters)",
            data: clusteredMeans,
            backgroundColor: "rgba(249, 115, 22, 0.8)",
            borderColor: "#f97316",
            borderWidth: 1,
            borderRadius: 6,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 1.0,
            max: 5.0,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            title: { display: true, text: "Mean Opinion Score (1–5)" }
          },
          x: {
            grid: { display: false }
          }
        },
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(3)} MOS`
            }
          }
        }
      }
    });

    // 6. Preference Doughnut Chart
    const ctxDoughnut = document.getElementById("prefDoughnutChart").getContext("2d");
    new Chart(ctxDoughnut, {
      type: "doughnut",
      data: {
        labels: [
          `Baseline Preferred (${pctBase}%)`,
          `Clustered Preferred (${pctClust}%)`,
          `No Preference (${pctNone}%)`
        ],
        datasets: [{
          data: [prefs.baseline, prefs.clustered, prefs.none],
          backgroundColor: [
            "rgba(99, 102, 241, 0.85)",
            "rgba(249, 115, 22, 0.85)",
            "rgba(148, 163, 184, 0.6)"
          ],
          borderColor: "#0b0f19",
          borderWidth: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" }
        },
        cutout: "65%",
      }
    });

  } catch (err) {
    console.error("Error rendering results dashboard", err);
  }
});
