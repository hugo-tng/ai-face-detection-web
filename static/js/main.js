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
