from django.shortcuts import render
from django.apps import apps
from PIL import Image
import io
import base64


def detect_view(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Đọc ảnh gốc từ memory
        image_bytes = image_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Encode ảnh gốc để preview
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        original_preview = base64.b64encode(buffer.getvalue()).decode()

        predictor = apps.get_app_config("detector").predictor

        # Inference (KHÔNG LƯU FILE)
        result = predictor.predict_from_pil(pil_image)

        context.update({
            "result": result,
            "original_preview": original_preview,
        })

    return render(request, "detect.html", context)
