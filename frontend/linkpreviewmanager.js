// Check if LinkPreviewManager already exists to avoid redefining
if (!window.LinkPreviewManager) {

  // if undefined redefine here 👇
  class LinkPreviewManager {
    constructor() {
      this.tooltip = null;
      this.currentLink = null;
      this.hoverTimeout = null;
      this.citationData = {};
    }

    registerCitation(citationId, data) {
      this.citationData[citationId] = data;
    }

    getLinkType(url, linkElement) {
      if (linkElement.dataset.citation || url.startsWith('#cite-')) {
        return 'citation';
      }
      
      try {
        const urlObj = new URL(url);
        if (urlObj.hostname) {
          return 'external';
        }
      } catch (e) {
        // Invalid URL
      }
      
      return 'internal';
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
      const link = event.target;
      console.log('Mouse entered link:', link.href);  // Add this debug line

      this.hoverTimeout = setTimeout(async () => {
        console.log('Creating tooltip...');
        this.currentLink = link;
        const url = link.href;
        const linkType = this.getLinkType(url, link);

        this.tooltip = this.createTooltip();
        this.tooltip.innerHTML = this.showLoading();

        const body = wwLib.getFrontDocument().body;
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
      const container = wwLib.getFrontDocument().querySelector(containerSelector);
      if (!container) {
        console.error('Container not found:', containerSelector);
        return;
      }
      
      const links = container.querySelectorAll('a');
      console.log(`Initializing ${links.length} links in ${containerSelector}`);
      
      links.forEach(link => {
        link.removeEventListener('mouseenter', this.boundHandleEnter);
        link.removeEventListener('mouseleave', this.boundHandleLeave);
        link.addEventListener('mouseenter', this.boundHandleEnter);
        link.addEventListener('mouseleave', this.boundHandleLeave);
      });
    }

    setup() {
      this.boundHandleEnter = this.handleMouseEnter.bind(this);
      this.boundHandleLeave = this.handleMouseLeave.bind(this);
    }
  }

  // Store the class globally
  window.LinkPreviewManager = LinkPreviewManager;
  console.log('LinkPreviewManager class defined');
}

// Initialize the manager
if (!window.linkPreviewManager) {
  window.linkPreviewManager = new window.LinkPreviewManager();
  window.linkPreviewManager.setup();
  console.log('Link Preview Manager initialized');
}

// Register test citations
window.linkPreviewManager.registerCitation('cite-1', {
  source: 'Nature Journal, 2024',
  text: 'Recent studies have shown that markdown-it provides excellent performance for real-time text rendering in web applications.',
  author: 'Smith et al.',
  date: '2024'
});

window.linkPreviewManager.registerCitation('cite-2', {
  source: 'Journal of Web Development, 2024',
  text: 'Performance benchmarks indicate that markdown parsing can be optimized significantly through proper implementation strategies.',
  author: 'Johnson & Lee',
  date: '2024'
});

window.linkPreviewManager.registerCitation('cite-3', {
  source: 'Computer Science Review, 2023',
  text: 'The DOM manipulation techniques used in modern JavaScript frameworks have evolved considerably.',
  author: 'Davis et al.',
  date: '2023'
});

// // 3. Initialize with longer delay
// setTimeout(() => {
//   if (window.linkPreviewManager) {
//     window.linkPreviewManager.initializeLinks('#test-link-preview');
//     console.log('✅ Initialized - try hovering over a link now!');
//   }
// }, 500);