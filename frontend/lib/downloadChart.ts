export const downloadChart = (containerId: string, filename: string) => {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container not found: ${containerId}`);
        return;
    }

    // Find SVG. Recharts wraps it, but querySelector should find it.
    // CRITICAL FIX: Ignore the download button icon which is also an SVG!
    const svgs = container.querySelectorAll('svg');
    let svg: SVGSVGElement | null = null;

    // Strategy: Find the largest SVG or specific class
    let maxSize = 0;

    for (let i = 0; i < svgs.length; i++) {
        const s = svgs[i];
        // Skip if inside a button
        if (s.closest('button')) continue;

        // Check specific class from Recharts
        if (s.classList.contains('recharts-surface')) {
            svg = s as SVGSVGElement;
            break;
        }

        // Fallback: Find largest by area
        const { width, height } = s.getBoundingClientRect();
        const area = width * height;
        if (area > maxSize && area > 1000) { // arbitrary small threshold (32x32=1024)
            maxSize = area;
            svg = s as SVGSVGElement;
        }
    }

    if (!svg) {
        console.error('Chart SVG not found in container (ignored icons)');
        return;
    }

    // Get computed dimensions
    const { width, height } = svg.getBoundingClientRect();

    if (width === 0 || height === 0) {
        console.error('SVG has 0 dimension');
        return;
    }

    // Clone bytes to avoid mutating original
    const clonedSvg = svg.cloneNode(true) as SVGElement;

    // Explicitly set width/height attributes if missing (Recharts uses 100% often)
    clonedSvg.setAttribute('width', width.toString());
    clonedSvg.setAttribute('height', height.toString());

    // Add white background rect if transparent (optional, but good for PNG)
    // Actually, we do this in canvas.

    const svgData = new XMLSerializer().serializeToString(clonedSvg);

    // Create a Blob
    const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    const img = new Image();

    img.onload = () => {
        const canvas = document.createElement('canvas');
        // High res export
        const scale = 2;
        canvas.width = width * scale;
        canvas.height = height * scale;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Fill background - Slate 900 to match theme
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw image
        ctx.scale(scale, scale);
        ctx.drawImage(img, 0, 0, width, height);

        // Download
        try {
            const pngUrl = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.download = `${filename}.png`;
            link.href = pngUrl;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (e) {
            console.error('Download failed', e);
        } finally {
            URL.revokeObjectURL(url);
        }
    };

    img.onerror = (e) => {
        console.error('Image load failed', e);
        URL.revokeObjectURL(url);
    };

    img.src = url;
};
