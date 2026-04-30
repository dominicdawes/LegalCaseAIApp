// Javascript action to add CSS dynamically if not already added (done on pageLoad or in a Workflow)
if (!document.getElementById('link-preview-styles')) {
  const style = document.createElement('style');
  style.id = 'link-preview-styles';
  style.textContent = `
    .link-preview-tooltip {
      position: fixed;
      background: white;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 0;
      box-shadow: 0 4px 16px rgba(0,0,0,0.12);
      max-width: 400px;
      z-index: 10000;
      opacity: 0;
      transition: opacity 0.2s ease;
      pointer-events: none;
    }

    .link-preview-tooltip.show {
      opacity: 1;
    }

    .link-preview-header {
      padding: 12px;
      border-bottom: 1px solid #f0f0f0;
    }

    .link-preview-title {
      font-weight: 600;
      font-size: 14px;
      margin-bottom: 4px;
      color: #1a1a1a;
    }

    .link-preview-url {
      font-size: 12px;
      color: #666;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .link-preview-body {
      padding: 12px;
    }

    .link-preview-description {
      font-size: 13px;
      line-height: 1.4;
      color: #333;
    }

    .link-preview-image {
      width: 100%;
      height: auto;
      max-height: 200px;
      object-fit: cover;
      border-radius: 4px;
      margin-top: 8px;
    }

    .link-preview-loading {
      padding: 12px;
      text-align: center;
      color: #999;
      font-size: 13px;
    }

    .citation-preview {
      background: #f8f9fa;
      border-left: 3px solid #4285f4;
      padding: 12px;
      font-size: 13px;
      line-height: 1.5;
      color: #333;
    }

    .citation-preview-source {
      font-weight: 600;
      margin-bottom: 6px;
      color: #1a1a1a;
    }
  `;
  document.head.appendChild(style);
  console.log('Link preview CSS injected');
}