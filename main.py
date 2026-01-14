# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from colour import extract_dominant_colours

app = FastAPI(title="Dominant Colour API")

# フロント（Vercel）から叩けるようにCORS許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番ではドメイン限定可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    mode: str = Form("auto"),  # "auto" or "manual"
    k: int = Form(5)
):
    # 画像を numpy 配列に変換
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    result = extract_dominant_colours(
        img_bgr=img,
        mode=mode,
        k=k
    )

    return result
