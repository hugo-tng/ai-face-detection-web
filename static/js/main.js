function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        let img = document.querySelector(".image-box img");
        if (!img) {
            img = document.createElement("img");
            document.querySelector(".image-box")?.appendChild(img);
        }
        img.src = reader.result;
    };
    reader.readAsDataURL(event.target.files[0]);
}

// Drag and drop
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileNameText = document.getElementById("file-name");

// Click → mở file picker
dropZone.addEventListener("click", () => {
    fileInput.click();
});

// Khi chọn file
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        fileNameText.textContent = fileInput.files[0].name;
    }
});

// Drag over
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("bg-light");
});

// Drag leave
dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("bg-light");
});

// Drop
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("bg-light");

    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileNameText.textContent = e.dataTransfer.files[0].name;
    }
});