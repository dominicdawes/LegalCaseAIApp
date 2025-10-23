// --- RUN THIS SCRIPT ON PAGE LOAD ---

(async () => {

  // --- Handle the Container Selector ---
  let containerSelector; // <-- CHANGED: Declare variable in the main scope
  if (typeof HTML_CONTAINER_SELECTOR !== "undefined") {
    // Use the globally defined variable (e.g., from WeWeb)
    // containerSelector = '#citation-preview-style';  // <-- larger div container element
    // containerSelector = '#rtf-link-prev';   // <-- specific rtf element
    containerSelector = variables[/* citationCssId */'1a1d9dd4-7046-4866-a782-9edef5b000f3']
    console.log(`Using predefined HTML_CONTAINER_SELECTOR: ${containerSelector}`);
  } else {
    // Use a default and warn...
    // containerSelector = '#citation-preview-style';  // <-- larger div container element
    // containerSelector = '#rtf-link-prev';   // <-- specific rtf element
    containerSelector = variables[/* citationCssId */'1a1d9dd4-7046-4866-a782-9edef5b000f3']
    console.warn(`HTML_CONTAINER_SELECTOR is missing. Using default: ${containerSelector}`);
  }

  // Check if LinkPreviewManager already exists to avoid redefining
  if (!window.LinkPreviewManager) {
    class LinkPreviewManager {
      constructor() {
        this.tooltip = null;
        this.currentLink = null;
        this.hoverTimeout = null;
        this.citationData = {}; // This will store our citation info
      }

      // --- WeWeb-safe DOM access ---
      getDocument() {
        return wwLib.getFrontDocument ? wwLib.getFrontDocument() : document;
      }
      getBody() {
        return this.getDocument().body;
      }

      registerCitation(citationId, data) {
        // We strip the hash for consistent lookups
        const cleanId = citationId.replace('#', '');
        this.citationData[cleanId] = data;
        console.log(`Found & Registered citation: ${cleanId}`, data);
      }

      getLinkType(url, linkElement) {
        // Check for citation link (e.g., <a href="#cite-...">)
        if (linkElement.getAttribute('href') && linkElement.getAttribute('href').startsWith('#cite-')) {
          return 'citation';
        }
        // Check for external link
        try {
          const urlObj = new URL(url, window.location.origin);
          if (urlObj.hostname !== window.location.hostname) {
            return 'external';
          }
        } catch (e) { }
        return 'internal'; // Default to internal
      }

      async fetchUrlPreview(url) {
        try {
          const response = await fetch(`https://api.microlink.io?url=${encodeURIComponent(url)}`);
          const data = await response.json();
          
          if (data.status === 'success') {
            return {
              title: data.data.title || 'No title',
              description: data.data.description || '',
              image: data.data.image?.url || data.data.logo?.url,
              url: data.data.url
            };
          }
        } catch (error) {
          console.error('Error fetching preview:', error);
        }
        
        return {
          title: 'Link Preview',
          description: url,
          url: url
        };
      }

      getCitationPreview(citationId) {
        const data = this.citationData[citationId];
        if (data) {
          return data;
        }
        
        return {
          source: 'Citation',
          text: `Reference: ${citationId}`,
          author: '',
          date: ''
        };
      }

      createTooltip() {
        const tooltip = wwLib.getFrontDocument().createElement('div');
        // const tooltip = this.getDocument().createElement('div') // <-- may be more WeWeb safe
        tooltip.className = 'link-preview-tooltip';
        return tooltip;
      }

      renderUrlPreview(previewData) {
        return `
          <div class="link-preview-header">
            <div class="link-preview-title">${this.escapeHtml(previewData.title)}</div>
            <div class="link-preview-url">${this.escapeHtml(previewData.url)}</div>
          </div>
          <div class="link-preview-body">
            ${previewData.description ? `<div class="link-preview-description">${this.escapeHtml(previewData.description)}</div>` : ''}
            ${previewData.image ? `<img src="${previewData.image}" class="link-preview-image" alt="Preview">` : ''}
          </div>
        `;
      }

      renderCitationPreview(citationData) {
        return `
          <div class="citation-preview">
            <div class="citation-preview-source">${this.escapeHtml(citationData.source)}</div>
            <div>${this.escapeHtml(citationData.text)}</div>
            ${citationData.author ? `<div style="margin-top: 8px; font-size: 12px; color: #666;">${this.escapeHtml(citationData.author)}${citationData.date ? `, ${citationData.date}` : ''}</div>` : ''}
          </div>
        `;
      }

      showLoading() {
        return '<div class="link-preview-loading">Loading preview...</div>';
      }

      escapeHtml(text) {
        const div = wwLib.getFrontDocument().createElement('div');
        // const div = this.getDocument().createElement('div'); // <--- may be more WeWeb safe
        div.textContent = text;
        return div.innerHTML;
      }

      positionTooltip(linkElement, tooltip) {
        const rect = linkElement.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let left = rect.left;
        let top = rect.bottom + 8;
        
        if (left + tooltipRect.width > viewportWidth - 20) {
          left = viewportWidth - tooltipRect.width - 20;
        }
        if (left < 20) {
          left = 20;
        }
        
        if (top + tooltipRect.height > viewportHeight - 20) {
          top = rect.top - tooltipRect.height - 8;
        }
        
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
      }

      async handleMouseEnter(event) {
        const link = event.target.closest('a'); // Find the nearest <a> tag parent

        // If you're not hovering over a link (or one of its children), do nothing.
        if (!link) {
          return;
        }
        // const link = event.target;
        console.log('🐁 Mouse entered link:', link.href);  // Add this debug line

        this.hoverTimeout = setTimeout(async () => {
          console.log('Creating tooltip...');
          this.currentLink = link;
          const url = link.href;
          const linkType = this.getLinkType(url, link);

          this.tooltip = this.createTooltip();
          this.tooltip.innerHTML = this.showLoading();

          const body = wwLib.getFrontDocument().body;
          // const body = this.getBody(); // <- may need to use WeWeb-safe body
          console.log('Appending tooltip to body:', body);
          body.appendChild(this.tooltip);  // ✅ Only append ONCE

          console.log('Tooltip created:', this.tooltip);
          console.log('Tooltip classes:', this.tooltip.className);

          this.positionTooltip(link, this.tooltip);
          setTimeout(() => {
            this.tooltip.classList.add('show');
            console.log('Tooltip should now be visible');
          }, 10);

          if (linkType === 'citation') {
            // const citationId = link.hash.replace('#', ''); // Get ID from hash
            const citationId = link.dataset.citation || link.hash.replace('#cite-', '');
            const citationData = this.getCitationPreview(citationId);
            this.tooltip.innerHTML = this.renderCitationPreview(citationData);
          } else if (linkType === 'external') {
            const previewData = await this.fetchUrlPreview(url);
            if (this.currentLink === link && this.tooltip) {
              this.tooltip.innerHTML = this.renderUrlPreview(previewData);
              this.positionTooltip(link, this.tooltip);
            }
          } else {
            this.tooltip.innerHTML = this.renderUrlPreview({
              title: 'Internal Link',
              description: link.textContent,
              url: url
            });
          }
        }, 300);
      }

      handleMouseLeave() {
        clearTimeout(this.hoverTimeout);
        
        if (this.tooltip) {
          this.tooltip.classList.remove('show');
          setTimeout(() => {
            if (this.tooltip && this.tooltip.parentNode) {
              this.tooltip.parentNode.removeChild(this.tooltip);
            }
            this.tooltip = null;
          }, 200);
        }
        
        this.currentLink = null;
      }

      initializeLinks(containerSelector = 'body') {
        const container = this.getDocument().querySelector(containerSelector); // <-- Use WeWeb-safe document
        // const container = wwLib.getFrontDocument().querySelector(containerSelector);
        if (!container) {
          // Gracefully warn and skip if the element isn't found
          console.warn(`Link preview container NOT found (${containerSelector}). Skipping initialization ⚠️`);
          return;
        } else {
          console.info(`🔎 FOUND LinkPreview containerHTMPId (${containerSelector})!`);
        }

        // Use event delegation on the container
        // This is MUCH more efficient and works for streaming content
        container.removeEventListener('mouseenter', this.boundHandleEnter, true); // Remove old listener
        container.addEventListener('mouseenter', this.boundHandleEnter, true); // Add new one

        container.removeEventListener('mouseleave', this.boundHandleLeave, true);
        container.addEventListener('mouseleave', this.boundHandleLeave, true);

        console.log(`Link previews INITIALIZED on containerHtmlId: ${containerSelector}`);
      }

      setup() {
        this.boundHandleEnter = this.handleMouseEnter.bind(this);
        this.boundHandleLeave = this.handleMouseLeave.bind(this);
      }
    } // End of LinkPreviewManager class

    // Store the class globally
    window.LinkPreviewManager = LinkPreviewManager;
    console.log(`LinkPreviewManager class defined: ${containerSelector}`);
  }

  // =============== Initialize the manager (once globally) ============================
  if (!window.linkPreviewManager) {
    window.linkPreviewManager = new window.LinkPreviewManager();
    window.linkPreviewManager.setup();
    console.log('*NEW* Link Preview Manager initialized ✅');
  } else {
    console.log('Link Preview Manager already set in browser... 😪');
  }

  // =============== INITIALIZE LINKS ONCE AT THE START ================================
  // We add a delay to ensure your chat container element exists on the page
  setTimeout(() => {
    if (window.linkPreviewManager) {
      window.linkPreviewManager.initializeLinks(containerSelector);
    }
  }, 1000); // 1 second delay

  // // =============== Register test citations (ONLY FOR DEBUG) ==========================

  // window.linkPreviewManager.registerCitation('cite-1', {
  //   source: 'Nature Journal, 2024',
  //   text: 'Recent studies have shown that markdown-it provides excellent performance for real-time text rendering in web applications.',
  //   author: 'Smith et al.',
  //   date: '2024'
  // });

  // window.linkPreviewManager.registerCitation('cite-2', {
  //   source: 'Journal of Web Development, 2024',
  //   text: 'Performance benchmarks indicate that markdown parsing can be optimized significantly through proper implementation strategies.',
  //   author: 'Johnson & Lee',
  //   date: '2024'
  // });

  // window.linkPreviewManager.registerCitation('cite-3', {
  //   source: 'Computer Science Review, 2023',
  //   text: 'The DOM manipulation techniques used in modern JavaScript frameworks have evolved considerably.',
  //   author: 'Davis et al.',
  //   date: '2023'
  // });

})(); // end of async block