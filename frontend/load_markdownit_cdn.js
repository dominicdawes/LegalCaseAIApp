// In your workflow Custom JS action import markdown it from cdn
if (!window.markdownit) {
  await new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/markdown-it@13.0.1/dist/markdown-it.min.js';
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

const md = window.markdownit();
const markdownText = variables['38b42fc8-e1ef-4f52-b7ef-34aa188d3d02'] // Your markdown source


const html = md.render(markdownText);

// // Update a variable with the HTML result
// wwLib.wwVariable.updateValue('html-output-variable-id', html);

return html