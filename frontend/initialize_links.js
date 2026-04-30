// 3. Initialize with longer delay
setTimeout(() => {
  if (window.linkPreviewManager) {
    window.linkPreviewManager.initializeLinks('#test-link-preview');
    console.log('✅ Initialized - try hovering over a link now!');
  }
}, 500);