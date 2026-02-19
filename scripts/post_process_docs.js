const fs = require('fs');
const path = require('path');

const DOCS_ROOT = path.join(__dirname, '..', 'docs');
const BASE_URL = 'https://vectorforgeml.work.gd';

function getRecFiles(dir) {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        if (stat && stat.isDirectory()) {
            results = results.concat(getRecFiles(filePath));
        } else if (file.endsWith('.html')) {
            results.push(filePath);
        }
    });
    return results;
}

const allHtmlFiles = getRecFiles(DOCS_ROOT);
const sitemapUrls = [];

allHtmlFiles.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');

    // 1. Calculate relative path to root
    const relPath = path.relative(DOCS_ROOT, filePath);
    const depth = relPath.split(path.sep).length - 1;
    let prefix = '../'.repeat(depth);
    if (prefix === '') prefix = './';

    // 2. Fix Assets Paths (CSS, JS, Images, Favicon)
    // Replace absolute /assets/ with relative
    content = content.replace(/href="\/assets\//g, `href="${prefix}assets/`);
    content = content.replace(/src="\/assets\//g, `src="${prefix}assets/`);
    content = content.replace(/href="\/docs\//g, `href="${prefix}docs/`); // Fix internal links if absolute
    content = content.replace(/href="\/install\//g, `href="${prefix}install/`);
    content = content.replace(/href="\/playground\//g, `href="${prefix}playground/`);

    // 3. Fix internal absolute links (e.g. href="/index.html")
    content = content.replace(/href="\/index.html"/g, `href="${prefix}index.html"`);

    // 4. Inject Favicon if missing (or fix it)
    if (!content.includes('rel="icon"')) {
        content = content.replace('</head>', `<link rel="icon" type="image/png" href="${prefix}assets/images/VectorForgeML_Logo.png">\n</head>`);
    }

    // 5. SEO & Meta Tags
    // Extract title
    const titleMatch = content.match(/<title>(.*?)<\/title>/);
    let pageTitle = titleMatch ? titleMatch[1] : 'VectorForgeML';

    // Standardize Title
    if (!pageTitle.includes('VectorForgeML')) {
        pageTitle = `${pageTitle} | VectorForgeML`;
        content = content.replace(/<title>.*?<\/title>/, `<title>${pageTitle}</title>`);
    }

    const urlPath = relPath.replace(/\\/g, '/');
    const fullUrl = `${BASE_URL}/${urlPath}`;
    sitemapUrls.push(fullUrl);

    // Meta Tags Injection
    const metaTags = `
    <!-- Global SEO -->
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="robots" content="index, follow">
    <meta name="author" content="Mohd Musheer">
    
    <!-- Open Graph -->
    <meta property="og:title" content="${pageTitle}">
    <meta property="og:type" content="website">
    <meta property="og:url" content="${fullUrl}">
    <meta property="og:image" content="${BASE_URL}/assets/images/VectorForgeML_Logo.png">
    
    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="${pageTitle}">
    <meta name="twitter:image" content="${BASE_URL}/assets/images/VectorForgeML_Logo.png">
    `;

    // Insert Meta Tags if not present (simple check)
    if (!content.includes('property="og:title"')) {
        content = content.replace('<head>', `<head>${metaTags}`);
    }

    // 6. Performance
    // Add defer to main.js if not present
    if (content.includes('src="' + prefix + 'assets/js/main.js"')) {
        content = content.replace(`src="${prefix}assets/js/main.js"`, `src="${prefix}assets/js/main.js" defer`);
    }

    // Lazy load images
    // Regex to find img tags without loading="lazy"
    content = content.replace(/<img((?!loading="lazy")[^>]+)>/g, '<img loading="lazy"$1>');

    // 7. Accessibility
    // Ensure html lang
    if (!content.includes('<html lang="en">')) {
        content = content.replace('<html>', '<html lang="en">');
    }

    fs.writeFileSync(filePath, content);
    console.log(`Optimized: ${relPath}`);
});

// Generate Sitemap
const sitemapContent = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapUrls.map(url => `  <url>
    <loc>${url}</loc>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>`).join('\n')}
</urlset>`;

fs.writeFileSync(path.join(DOCS_ROOT, 'sitemap.xml'), sitemapContent);
console.log('Generated sitemap.xml');

// Generate Robots.txt
const robotsContent = `User-agent: *
Allow: /
Sitemap: ${BASE_URL}/sitemap.xml`;

fs.writeFileSync(path.join(DOCS_ROOT, 'robots.txt'), robotsContent);
console.log('Generated robots.txt');
