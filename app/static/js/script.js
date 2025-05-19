document.addEventListener('DOMContentLoaded', () => {
  const uploadCard = document.querySelector('.upload-card');
  const imageUpload = document.getElementById('imageUpload');
  const imagePreview = document.getElementById('imagePreview');
  const resultsDiv = document.getElementById('results');
  const defectsList = document.getElementById('defectsList');
  const loader = document.querySelector('.loader');
  const btn = document.querySelector('.btn');

  // Drag and Drop эффекты
  uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.style.background = 'rgba(108, 99, 255, 0.1)';
    uploadCard.style.border = '2px dashed var(--primary-color)';
  });

  uploadCard.addEventListener('dragleave', () => {
    uploadCard.style.background = 'white';
    uploadCard.style.border = '2px dashed #ccc';
  });

  uploadCard.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadCard.style.background = 'white';
    uploadCard.style.border = '2px dashed #ccc';

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleImageUpload(file);
    }
  });

  // Обработка загрузки файла
  imageUpload.addEventListener('change', (e) => {
    if (e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  });

  // Отправка на сервер и отображение превью
  async function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      uploadCard.style.border = 'none';
    };
    reader.readAsDataURL(file);

    // Показываем загрузку
    loader.classList.add('active');
    btn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Ошибка при обработке изображения');

      const data = await response.json();
      displayResults(data);
    } catch (error) {
      alert(error.message);
    } finally {
      loader.classList.remove('active');
      btn.disabled = false;
    }
  }

  // Отображение результатов
  function displayResults(data) {
    defectsList.innerHTML = '';

    if (data.defects && data.defects.length > 0) {
      data.defects.forEach(defect => {
        const defectItem = document.createElement('div');
        defectItem.className = 'defect-item';
        defectItem.innerHTML = `
          <span><strong>Тип:</strong> ${defect.class}</span>
          <span><strong>Точность:</strong> ${defect.confidence}</span>
        `;
        defectsList.appendChild(defectItem);
      });
    } else {
      defectsList.innerHTML = '<div class="defect-item">Дефекты не обнаружены</div>';
    }

    // Анимация появления
    setTimeout(() => {
      resultsDiv.classList.add('show');
    }, 100);
  }
});