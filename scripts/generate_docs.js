const fs = require('fs');
const path = require('path');
const docsData = require('./docs_data.js');

// List of all algorithms to generate (merging keys from docsData and the original list if needed)
// For now, we rely on what's in docsData + any extras we simply list
const extraAlgorithms = [
    "drop_constant_columns", "find_best_k", "fit_linear_model",
    "macro_f1", "macro_precision", "macro_recall", "mse", "OneHotEncoder", "Pipeline",
    "plot_confusion_matrix", "precision_score", "predict_linear_model", "r2_score",
    "recall_score", "rmse", "train_test_split", "ColumnTransformer", "confusion_stats"
];

// Combine all unique keys
const allAlgos = [...new Set([...Object.keys(docsData), ...extraAlgorithms])];

const template = (algoName, data) => {
    const title = data ? data.title : algoName;
    const desc = data ? data.description : `Documentation for ${algoName}.`;

    // Conditional Sections
    const showSteps = data && data.steps && data.steps.length > 0;
    const showComplexity = data && data.time && data.time !== 'N/A';

    const stepsSection = showSteps ? `
            <section class="slide-up" style="animation-delay: 0.1s; margin-top: 3rem;">
                <h3>Algorithm Workflow</h3>
                <div class="workflow-container">
                    ${data.steps.map(s => {
        const parts = s.split(':');
        if (parts.length > 1 && /^[A-Z0-9 ]+$/.test(parts[0].trim())) {
            const key = parts[0].trim();
            const val = parts.slice(1).join(':').trim();
            return `<div class="workflow-step">
                                <span class="workflow-key">${key}</span>
                                <span class="workflow-val">${val}</span>
                            </div>`;
        } else {
            return `<div class="workflow-step"><span class="workflow-val" style="margin-left:0;">${s}</span></div>`;
        }
    }).join('')}
                </div>
            </section>` : '';

    const math = data && data.math ? `<div class="glass-panel" style="padding:1rem; margin:1rem 0; font-family:'Times New Roman', serif; font-size:1.1rem; text-align:center;">${data.math}</div>` : '';

    const showImpl = data && (data.impl || data.code);
    const implText = data && data.impl ? data.impl : 'Implemented in R/C++.';
    const lang = data && data.lang ? data.lang : 'cpp';
    const implCode = data && data.code ? `
                <div class="glass-panel" style="margin-top: 1.5rem; padding: 0; overflow: hidden;">
                    <pre><code class="language-${lang}" style="font-size: 0.85rem;">${data.code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
                </div>` : '';

    const implSection = showImpl ? `
            <section class="slide-up" style="animation-delay: 0.2s; margin-top: 3rem;">
                <h3>Implementation Details</h3>
                <div class="glass-panel" style="padding: 2rem;">
                    <p style="color: var(--text-secondary);">${implText}</p>
                </div>
                ${implCode}
            </section>` : '';

    const complexitySection = showComplexity ? `
            <section class="slide-up" style="animation-delay: 0.3s; margin-top: 3rem;">
                <h3>Complexity & Optimization</h3>
                <div class="grid-2">
                    <div class="glass-panel" style="padding: 1.5rem;">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-secondary);">Time Complexity</h4>
                        <p style="color: var(--text-secondary); margin: 0;">${data.time}</p>
                    </div>
                    <div class="glass-panel" style="padding: 1.5rem;">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-primary);">Space Complexity</h4>
                        <p style="color: var(--text-secondary); margin: 0;">${data.space}</p>
                    </div>
                    <div class="glass-panel" style="padding: 1.5rem;">
                        <h4 style="margin-bottom: 0.5rem; color: var(--text-primary);">Optimizations</h4>
                        <p style="color: var(--text-secondary); margin: 0;">${data.opt || 'None'}</p>
                    </div>
                    <div class="glass-panel" style="padding: 1.5rem;">
                        <h4 style="margin-bottom: 0.5rem; color: #ef4444;">Limitations</h4>
                        <p style="color: var(--text-secondary); margin: 0;">${data.limitations || 'None listed'}</p>
                    </div>
                </div>
            </section>` : '';

    const time = data && data.time ? data.time : 'N/A';
    const space = data && data.space ? data.space : 'N/A';
    const opt = data && data.opt ? data.opt : 'None';
    const cases = data && data.cases ? data.cases : 'General purpose.';
    const limitations = data && data.limitations ? data.limitations : 'None listed.';

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} - VectorForgeML Documentation</title>
    <meta name="description" content="${desc.substring(0, 160).replace(/"/g, '&quot;')}...">
    <meta name="keywords" content="${title}, ${algoName}, R Machine Learning, C++ Implementation, VectorForgeML, Data Science, Algorithm">
    <link rel="canonical" href="https://vectorforgeml.work.gd/docs/${algoName}.html">
    
    <!-- Open Graph -->
    <meta property="og:title" content="${title} - High Performance Implementation">
    <meta property="og:description" content="${desc.substring(0, 200).replace(/"/g, '&quot;')}...">
    <meta property="og:type" content="article">
    <meta property="og:url" content="https://vectorforgeml.work.gd/docs/${algoName}.html">
    <meta property="og:image" content="https://vectorforgeml.work.gd/assets/images/VectorForgeML_Logo.png">

    <link rel="icon" type="image/png" href="/assets/images/VectorForgeML_Logo.png">
    <link rel="stylesheet" href="/assets/css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code&display=swap" rel="stylesheet">
    
    <!-- MathJax -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <!-- Schema.org -->
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": "${title} Implementation in VectorForgeML",
      "description": "${desc.replace(/"/g, '&quot;')}",
      "proficiencyLevel": "Expert",
      "programmingLanguage": ["R", "C++"],
      "author": {
        "@type": "Person",
        "name": "Mohd Musheer"
      },
      "publisher": {
        "@type": "Organization",
        "name": "VectorForgeML",
        "logo": {
          "@type": "ImageObject",
          "url": "https://vectorforgeml.work.gd/assets/images/VectorForgeML_Logo.png"
        }
      }
    }
    </script>
</head>
<body class="docs-layout">
    <div id="navbar"></div>
    <div id="sidebar"></div>

    <div class="layout-container">
        <main class="main-content">
            <div style="margin-bottom: 2rem;">
                <span style="font-size: 0.9rem; color: var(--accent-primary); text-transform: uppercase; letter-spacing: 1px;">Algorithm</span>
                <h1 class="fade-in" style="margin-top: 0.5rem;">${title}</h1>
            </div>

            <section class="glass-panel slide-up" style="padding: 2rem; margin-bottom: 2rem;">
                <h3>Description</h3>
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    ${desc}
                </p>
                ${math}
            </section>

            ${stepsSection}

            ${implSection}

            ${complexitySection}

             <section class="slide-up" style="animation-delay: 0.4s; margin-top: 3rem;">
                <h3>Use Cases</h3>
                 <p style="color: var(--text-secondary);">${cases}</p>
            </section>

        </main>
    </div>

    <div id="footer"></div>
    <script type="module" src="/assets/js/main.js"></script>
</body>
</html>`;
};

const outputDir = path.join(__dirname, '..', 'public', 'docs');

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

allAlgos.forEach(algo => {
    const data = docsData[algo];
    fs.writeFileSync(path.join(outputDir, `${algo}.html`), template(algo, data));
    console.log(`Generated ${algo}.html`);
});
