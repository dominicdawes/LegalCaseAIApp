// Function to initialize Markdown-it and LinkPreview CSS (from weweb cdn)
async function initializeChatSystem() {

    // --- Load markdown-it ---
    if (!window.markdownit) {
        console.log('Loading markdown-it library...');
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/markdown-it@13.0.1/dist/markdown-it.min.js';
            script.onload = () => {
                console.log('markdown-it success ✅');
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    } else {
        console.log("Skipped re-download of markdown-it from CDN");
    }

    if (!window.markdownItAttrs) {
        console.log('Loading markdown-it-attrs library...');
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            // Use the correct browser bundle path
            script.src = 'https://unpkg.com/markdown-it-attrs@4/dist/markdown-it-attrs.browser.js';
            script.onload = () => {
                console.log('markdown-it-attr success ✅');
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    } else {
        console.log("Skipped re-download of markdown-it-attrs from CDN");
    }


    // Initialize markdown-it instance globally AND USE THE PLUGIN
    // We also check if window.md already exists to avoid re-initializing
    if (!window.md) {
        if (window.markdownit && window.markdownItAttrs) {
            window.md = window.markdownit({ html: true }).use(window.markdownItAttrs);
            console.log('markdown-it and markdown-it-attrs initialized ✅');
        } else {
            console.error('Failed to load markdown-it or markdown-it-attrs. Cannot initialize.');
        }
    } else {
        console.log('markdown-it instance (window.md) already exists.');
    }

    // --- Inject Link Preview CSS ---
    const STYLE_ID_TAG = 'citation-preview-style' // give the appropriate div the HTML id of: citation-preview-style

    if (!document.getElementById(STYLE_ID_TAG)) {
        const style = document.createElement('style');
        style.id = STYLE_ID_TAG;
        style.textContent = `
            .link-preview-tooltip { position: fixed; background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 0; box-shadow: 0 4px 16px rgba(0,0,0,0.12); max-width: 400px; z-index: 10000; opacity: 0; transition: opacity 0.2s ease; pointer-events: none; }
            .link-preview-tooltip.show { opacity: 1; }
            .link-preview-header { padding: 12px; border-bottom: 1px solid #f0f0f0; }
            .link-preview-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; color: #1a1a1a; }
            .link-preview-url { font-size: 12px; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .link-preview-body { padding: 12px; }
            .link-preview-description { font-size: 13px; line-height: 1.4; color: #333; }
            .link-preview-image { width: 100%; height: auto; max-height: 200px; object-fit: cover; border-radius: 4px; margin-top: 8px; }
            .link-preview-loading { padding: 12px; text-align: center; color: #999; font-size: 13px; }
            .citation-preview { background: #f8f9fa; border-left: 3px solid #4285f4; padding: 12px; font-size: 13px; line-height: 1.5; color: #333; }
            .citation-preview-source { font-weight: 600; margin-bottom: 6px; color: #1a1a1a; }
            .chat-citation-link { background-color: #e8f0fe; color: #1a73e8; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 500; text-decoration: none; white-space: nowrap; }
            .chat-citation-link:hover { background-color: #d2e3fc; }
        `;
        document.head.appendChild(style);
        console.log('🎨 Link preview CSS injected.');
    } else {
        console.log("CSS style tag already exists.");
    }

    console.log('CSS and Markdown-it initialized!!');
}

// Run the initialization
initializeChatSystem();