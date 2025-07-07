# Image Upscaling + Gemini OCR API (FastAPI)
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import google.generativeai as genai
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBOxvqR31egf1IhP2CRRUn8R2dzTkyawCo" 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def upscale_and_enhance_image_from_bytes(image_bytes, scale_factor=2):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled_img = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    pil_img = Image.fromarray(upscaled_img)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(3.3)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.5)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.1)

    cv_img = np.array(pil_img)
    denoised_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
    smoothed_img = cv2.bilateralFilter(denoised_img, d=9, sigmaColor=75, sigmaSpace=75)

    return Image.fromarray(smoothed_img)

def get_text_from_image(image: Image.Image, prompt: str = "You are an expert at reading and extracting text from complex, distorted images.The image I am providing is a CAPTCHA-style image used to prevent automated bots. It contains several alphanumeric characters that are: Capital letters and numbers only (A–Z, 0–9) Red-colored text on a white background Overlaid with multiple red distortion lines that may intersect with or pass through the charactersSlightly rotated, warped, or spaced unevenlyNot separated into clean segmentsPlease:Carefully ignore the background distortion and extract only the visible characters in their correct left-to-right order, as a human would read them.Return the final string of characters only, without any explanation or additional commentary.If any character is truly unreadable, mark it as [?] instead of guessing.Do not include any spaces or special characters unless they are clearly part of the CAPTCHA.") -> str:
    image = image.convert("RGB").resize((640, 480))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    response = model.generate_content([prompt, Image.open(buf)])
    return response.text.strip().upper()

@app.post("/api/extract-text")
async def extract_text(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        enhanced_image = upscale_and_enhance_image_from_bytes(image_bytes)
        extracted_text = get_text_from_image(enhanced_image)
        return JSONResponse(content={"text": extracted_text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
