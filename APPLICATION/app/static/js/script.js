document.addEventListener('DOMContentLoaded', () => {
    const uploadCard = document.getElementById('uploadCard');
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const resultsDiv = document.getElementById('results');
    const defectsList = document.getElementById('defectsList');
    const loader = document.getElementById('loader');
    const resultLink = document.getElementById('resultLink');
    const processedImageLink = document.getElementById('processedImageLink');

    // Drag and Drop
    uploadCard.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadCard.classList.add('dragover');
    });

    uploadCard.addEventListener('dragleave', () => {
        uploadCard.classList.remove('dragover');
    });

    uploadCard.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadCard.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File selection
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    async function handleFile(file) {
        // Check file type
        if (!file.type.match('image.*')) {
            alert('Пожалуйста, выберите изображение');
            return;
        }

        // Show preview
        imagePreview.src = URL.createObjectURL(file);
        imagePreview.style.display = 'block';

        // Reset previous results
        defectsList.innerHTML = '';
        resultLink.style.display = 'none';
        resultsDiv.style.display = 'none';

        // Show loader
        loader.style.display = 'flex';

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Send to predict endpoint
            const predictResponse = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!predictResponse.ok) {
                throw new Error(`Ошибка анализа: ${predictResponse.status}`);
            }

            const predictData = await predictResponse.json();
            displayDefects(predictData);

            // Send to upload endpoint
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (uploadResponse.ok) {
                const uploadData = await uploadResponse.json();
                if (uploadData.result_url) {
                    processedImageLink.href = uploadData.result_url;
                    resultLink.style.display = 'block';
                }
            }
            const reportResponse = await fetch('/report', {
                method: 'GET'
            });

            if (reportResponse.ok) {
                const reportData = await reportResponse.json();
                if (reportData.report_url) {
                    createDownloadLink(reportData.report_url, file.name);
                }
            }

        } catch (error) {
            console.error('Error:', error);
            defectsList.innerHTML = `
                <div class="error">
                    <p><strong>Ошибка:</strong> ${error.message}</p>
                </div>
            `;
            resultsDiv.style.display = 'block';
        } finally {
            loader.style.display = 'none';
        }
    }

    function createDownloadLink(reportUrl, filename = 'report') {
        const container = document.getElementById('downloadReportContainer');
        if (!container) {
            console.error('Контейнер downloadReportContainer не найден');
            return;
        }

        // Безопасная обработка имени файла
        const safeFilename = filename ? filename.replace(/\.[^/.]+$/, "") : 'report';
        const downloadName = `${safeFilename}_report.docx`;

        container.innerHTML = `
            <h3 style="margin-bottom: 15px;">Отчет по анализу</h3>
            <a href="${reportUrl}" download="${downloadName}" class="result-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Скачать отчет (Word)
            </a>
            <p style="margin-top: 10px; color: var(--gray); font-size: 14px;">
                Отчет содержит полную информацию о выявленных дефектах
            </p>
        `;
    }

    function displayDefects(results) {
        defectsList.innerHTML = '';

        if (!Array.isArray(results)) {
            defectsList.innerHTML = `
                <div class="error">
                    <p>Некорректный формат данных от сервера</p>
                </div>
            `;
            resultsDiv.style.display = 'block';
            return;
        }

        results.forEach((result, index) => {
            const partDiv = document.createElement('div');
            partDiv.className = 'part-result';

            const partTitle = document.createElement('h3');
            partTitle.className = 'part-title';

            // Add icon based on status
            const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            icon.setAttribute('width', '20');
            icon.setAttribute('height', '20');
            icon.setAttribute('viewBox', '0 0 24 24');
            icon.setAttribute('fill', 'none');
            icon.setAttribute('stroke', 'currentColor');
            icon.setAttribute('stroke-width', '2');
            icon.setAttribute('stroke-linecap', 'round');
            icon.setAttribute('stroke-linejoin', 'round');

            if (result.status === 'no_defects') {
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', 'M22 11.08V12a10 10 0 1 1-5.93-9.14');
                icon.appendChild(path);

                const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                polyline.setAttribute('points', '22 4 12 14.01 9 11.01');
                icon.appendChild(polyline);
            } else {
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', 'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z');
                icon.appendChild(path);

                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', '12');
                line.setAttribute('y1', '9');
                line.setAttribute('x2', '12');
                line.setAttribute('y2', '13');
                icon.appendChild(line);

                const line2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line2.setAttribute('x1', '12');
                line2.setAttribute('y1', '17');
                line2.setAttribute('x2', '12.01');
                line2.setAttribute('y2', '17');
                icon.appendChild(line2);
            }

            partTitle.appendChild(icon);
            partTitle.appendChild(document.createTextNode(
                `Область ${index + 1}: ${result.status === 'no_defects' ? 'Без дефектов' : 'Дефекты обнаружены'}`
            ));
            partDiv.appendChild(partTitle);

            if (result.defects && result.defects.length > 0) {
                result.defects.forEach((defect, i) => {
                    const defectItem = document.createElement('div');
                    defectItem.className = 'defect-item';

                    defectItem.innerHTML = `
                        <p class="defect-type"><strong>Дефект ${i + 1}:</strong> ${defect.class}</p>
                        <p class="defect-confidence"><strong>Уверенность:</strong> ${defect.confidence}</p>
                        <p class="defect-coordinates"><strong>Координаты:</strong> ${defect.coordinates}</p>
                        <p class="defect-length"><strong>Длина по линейке:</strong> ${defect.length}</p>
                    `;
                    partDiv.appendChild(defectItem);
                });
            } else {
                const noDefects = document.createElement('div');
                noDefects.className = 'no-defects';
                noDefects.textContent = 'Дефекты не обнаружены';
                partDiv.appendChild(noDefects);
            }

            defectsList.appendChild(partDiv);
        });

        resultsDiv.style.display = 'block';
    }
});