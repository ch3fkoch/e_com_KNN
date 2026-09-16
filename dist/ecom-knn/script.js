/**
 * E-Com KNN Project - Frontend Logic & Live ML Inference Engine
 * Autorin: Jane (Moderne Frontend- & UI-Systems-Entwicklerin)
 * ML-Architektur: Alex (Senior Python Engineer)
 * Standards: WCAG 2.2 AA Accessibility, localStorage Persistence, Real-Time Client Inference
 */

(function () {
    'use strict';

    // =========================================================================
    // 1. THEME MANAGEMENT (Light / Dark Mode)
    // =========================================================================
    const STORAGE_KEY = 'ecom_knn_theme';
    const THEME_ATTR = 'data-theme';
    const toggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    function getPreferredTheme() {
        const savedTheme = localStorage.getItem(STORAGE_KEY);
        if (savedTheme) {
            return savedTheme;
        }
        return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    function setTheme(theme) {
        document.documentElement.setAttribute(THEME_ATTR, theme);
        localStorage.setItem(STORAGE_KEY, theme);

        const isLight = theme === 'light';
        if (toggleBtn) {
            toggleBtn.setAttribute('aria-pressed', isLight ? 'true' : 'false');
            toggleBtn.setAttribute('aria-label', isLight ? 'Zu dunklem Modus wechseln' : 'Zu hellem Modus wechseln');
        }

        if (themeIcon) {
            themeIcon.innerHTML = isLight
                ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
                : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
        }

        if (themeText) {
            themeText.textContent = isLight ? 'Light' : 'Dark';
        }
    }

    const initialTheme = getPreferredTheme();
    setTheme(initialTheme);

    if (toggleBtn) {
        toggleBtn.addEventListener('click', function () {
            const currentTheme = document.documentElement.getAttribute(THEME_ATTR) || 'dark';
            const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(nextTheme);
        });
    }

    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function (e) {
        if (!localStorage.getItem(STORAGE_KEY)) {
            setTheme(e.matches ? 'light' : 'dark');
        }
    });

    // =========================================================================
    // 2. LIVE ML INFERENCE ENGINE (Echtes trainiertes Multi-Layer Perceptron)
    // =========================================================================
    // Skalierungs-Parameter (StandardScaler) und Gewichte aus Scikit-Learn (12.330 Sessions)
    const MODEL = {
        scaler_mean: [5.88925786, 0.04307280, 1194.74622],
        scaler_scale: [18.56768361, 0.04859457, 1913.59168],
        W1: [
            [-0.6167132, 0.5947314, 0.4822897, 0.2037998, -1.0849974, -0.1239055, -1.0695854, 0.3681064, 0.0493682, 0.2075658, -1.5098176, 0.5784114, 0.2970285, -1.1154062, -0.9898362, -0.3900199],
            [0.1481525, 0.2817467, -0.4222549, -0.9285771, 0.5052191, -0.6043580, -0.0333366, -0.1885264, -0.0396180, 0.0380522, -0.6134640, -0.4393158, 0.3348543, -0.2586065, 0.2829367, -0.4129177],
            [-0.4818330, 0.4108218, 0.2498565, 0.2292626, -0.0783770, -0.6429974, 0.1064566, -0.2493284, -0.3171426, -1.1995019, -0.3957960, 0.3351319, -0.0424690, -0.0150675, -0.0098689, 0.1183684]
        ],
        b1: [-0.0474655, -0.0743204, 0.6090309, 0.3830851, 0.3132445, 0.2787387, 0.0736204, 0.6533679, -0.5205586, -0.0198994, -0.4293420, 0.1613700, -0.2592281, -0.2747714, -0.0077663, -0.3219404],
        W2: [
            [-0.4426131, 0.2472099, -0.2546429, 0.0, -0.1145638, 0.8692497, 0.4920625, -0.4925124],
            [-0.4427612, 0.2871307, 0.2439636, 0.0, 0.2841778, -0.3670389, -0.2072106, -0.3680342],
            [0.4721272, 0.0450853, -0.1561448, 0.0, -0.0523523, -0.4107846, 0.1377567, 0.2735376],
            [0.5796998, -0.1807983, -0.4351120, 0.0, 0.4927065, -0.1165782, 0.1012502, 0.2331033],
            [-0.1770809, 0.0952270, -0.2005796, 0.0, -0.6028331, 0.4474257, 0.0060799, -0.2330766],
            [0.5294453, -0.3120696, -0.4612247, 0.0, 0.2599127, -0.2537882, -0.2270556, -0.0850751],
            [0.1641464, 0.5783407, 0.1866064, 0.0, 0.1918211, 0.2402243, 0.6381722, -0.2102821],
            [0.4156679, 0.3223655, -0.0748510, 0.0, -0.0911226, -0.3328431, 0.2756488, 0.5216725],
            [-0.2620627, -0.1075118, -0.1982961, 0.0, -0.1916686, -0.1705804, 0.4088616, -0.0184580],
            [0.5141961, -0.2636228, -0.0886283, 0.0, 0.9829817, -0.4979630, -0.5151782, 0.2942312],
            [-0.9474047, 0.2317214, 0.0, 0.0, -1.0970271, 0.6660577, 1.1197675, -0.8471288],
            [-0.2378690, -0.0965156, 0.5087297, 0.0, 0.2816048, -0.0468494, -0.3500591, 0.3325133],
            [-0.0717117, 0.0542972, 0.2596388, 0.0, -0.3136517, 0.1628343, -0.3471573, -0.2508424],
            [-0.8858412, 0.5123683, 0.3202330, 0.0, -0.2959306, 0.4443865, 0.5746661, -0.6482852],
            [-0.0742675, 0.1119243, 0.4614066, 0.0, -0.3643407, -0.0360226, 0.6730605, 0.0954434],
            [-0.4260380, 0.4316244, 0.4378790, 0.0, -0.2120835, 0.4732083, -0.2398860, 0.1181628]
        ],
        b2: [0.2440240, 0.3098773, -0.1429037, -0.1507904, 0.4173222, 0.4818641, 0.5671196, 0.2133310],
        W3: [
            [0.4040091],
            [-0.8418729],
            [-0.6693404],
            [0.0],
            [0.5017633],
            [-1.3728730],
            [-1.0509603],
            [0.4762412]
        ],
        b3: [-0.9647583]
    };

    function relu(x) {
        return Math.max(0, x);
    }

    function sigmoid(x) {
        const clamped = Math.max(-50, Math.min(50, x));
        return 1 / (1 + Math.exp(-clamped));
    }

    /**
     * Führt die Vorhersage im MLP-Netz durch
     */
    function predictMLP(pageValues, exitRate, duration) {
        const t0 = performance.now();

        // 1. StandardScaler
        const x0 = (pageValues - MODEL.scaler_mean[0]) / MODEL.scaler_scale[0];
        const x1 = (exitRate - MODEL.scaler_mean[1]) / MODEL.scaler_scale[1];
        const x2 = (duration - MODEL.scaler_mean[2]) / MODEL.scaler_scale[2];
        const input = [x0, x1, x2];

        // 2. Hidden Layer 1 (16 Neuronen, ReLU)
        const h1 = new Array(16);
        for (let j = 0; j < 16; j++) {
            let sum = MODEL.b1[j];
            sum += input[0] * MODEL.W1[0][j];
            sum += input[1] * MODEL.W1[1][j];
            sum += input[2] * MODEL.W1[2][j];
            h1[j] = relu(sum);
        }

        // 3. Hidden Layer 2 (8 Neuronen, ReLU)
        const h2 = new Array(8);
        for (let k = 0; k < 8; k++) {
            let sum = MODEL.b2[k];
            for (let j = 0; j < 16; j++) {
                sum += h1[j] * MODEL.W2[j][k];
            }
            h2[k] = relu(sum);
        }

        // 4. Output Layer (1 Neuron, Sigmoid)
        let out = MODEL.b3[0];
        for (let k = 0; k < 8; k++) {
            out += h2[k] * MODEL.W3[k][0];
        }

        const prob = sigmoid(out);
        const latencyMs = (performance.now() - t0).toFixed(2);

        return { prob, latencyMs };
    }

    // UI-Elemente des Simulators
    const sliderPageValues = document.getElementById('slider-pagevalues');
    const sliderExitRate = document.getElementById('slider-exitrate');
    const sliderDuration = document.getElementById('slider-duration');

    const valPageValues = document.getElementById('val-pagevalues');
    const valExitRate = document.getElementById('val-exitrate');
    const valDuration = document.getElementById('val-duration');

    const gaugePercent = document.getElementById('gauge-percent');
    const gaugeFill = document.getElementById('gauge-fill');
    const actionBadge = document.getElementById('action-badge');
    const actionTitle = document.getElementById('action-title');
    const actionDesc = document.getElementById('action-desc');
    const latencyBadge = document.getElementById('latency-badge');

    function updateSimulator() {
        if (!sliderPageValues || !sliderExitRate || !sliderDuration) return;

        const pv = parseFloat(sliderPageValues.value);
        const er = parseFloat(sliderExitRate.value);
        const dur = parseFloat(sliderDuration.value);

        if (valPageValues) valPageValues.textContent = pv.toFixed(1) + ' €';
        if (valExitRate) valExitRate.textContent = (er * 100).toFixed(1) + ' %';
        if (valDuration) valDuration.textContent = Math.round(dur) + ' s';

        const { prob, latencyMs } = predictMLP(pv, er, dur);
        const pct = (prob * 100).toFixed(1);

        if (gaugePercent) gaugePercent.textContent = pct + ' %';
        if (gaugeFill) gaugeFill.style.width = pct + '%';
        if (latencyBadge) latencyBadge.textContent = `${latencyMs} ms`;

        // Business Decision Logic
        if (prob < 0.40) {
            // Niedrige Kaufabsicht -> Kein Anreiz
            if (actionBadge) {
                actionBadge.className = 'status-badge status-low';
                actionBadge.textContent = 'Geringe Kaufabsicht';
            }
            if (actionTitle) actionTitle.textContent = 'Keine Aktion erforderlich';
            if (actionDesc) actionDesc.textContent = 'Besucher ist im Erkundungsmodus. Kein Rabattbudget verschwenden.';
            if (gaugeFill) gaugeFill.style.background = 'var(--text-muted)';
        } else if (prob >= 0.40 && prob <= 0.60) {
            // Wackelkandidat -> SMART VOUCHER
            if (actionBadge) {
                actionBadge.className = 'status-badge status-target';
                actionBadge.textContent = 'Wackelkandidat (40–60%)';
            }
            if (actionTitle) actionTitle.textContent = '⚡ SMART VOUCHER AKTIVIERT';
            if (actionDesc) actionDesc.textContent = 'Gezielter 10%-Gutschein ausgespielt: Maximale Conversion bei 0% Mitnahmeeffekt!';
            if (gaugeFill) gaugeFill.style.background = 'var(--brand-accent)';
        } else {
            // Hohe Kaufabsicht -> Margenschutz
            if (actionBadge) {
                actionBadge.className = 'status-badge status-high';
                actionBadge.textContent = 'Hohe Kaufabsicht (>60%)';
            }
            if (actionTitle) actionTitle.textContent = 'Margenschutz aktiv (Kein Rabatt)';
            if (actionDesc) actionDesc.textContent = 'Kaufbereiter Kunde schließt den Kauf selbstständig ab. Marge geschützt.';
            if (gaugeFill) gaugeFill.style.background = 'var(--brand-cyan)';
        }
    }

    if (sliderPageValues && sliderExitRate && sliderDuration) {
        sliderPageValues.addEventListener('input', updateSimulator);
        sliderExitRate.addEventListener('input', updateSimulator);
        sliderDuration.addEventListener('input', updateSimulator);
        // Initialer Durchlauf
        updateSimulator();
    }

    // =========================================================================
    // 3. LIGHTBOX MODAL
    // =========================================================================
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalCaption = document.getElementById('modal-caption');
    const modalClose = document.getElementById('modal-close');

    if (modal && modalImg && modalClose) {
        document.querySelectorAll('.card-image-wrapper img').forEach(function (img) {
            img.addEventListener('click', function () {
                modal.removeAttribute('hidden');
                modalImg.src = this.src;
                modalImg.alt = this.alt;
                if (modalCaption) {
                    modalCaption.textContent = this.alt;
                }
                modalClose.focus();
                document.body.style.overflow = 'hidden';
            });
        });

        function closeModal() {
            modal.setAttribute('hidden', '');
            modalImg.src = '';
            document.body.style.overflow = '';
        }

        modalClose.addEventListener('click', closeModal);
        modal.addEventListener('click', function (e) {
            if (e.target === modal) {
                closeModal();
            }
        });

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && !modal.hasAttribute('hidden')) {
                closeModal();
            }
        });
    }
})();
