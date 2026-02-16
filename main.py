# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from colour import extract_dominant_colours
from typing import Optional
import traceback

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
    k: Optional[str] = Form(None)
):
    try:
        # kをパース（空文字やnoneはnone扱い）
        k_int: Optional[int] = None
        if k is not None:
            k = k.strip()
            if k != "":
                try: 
                    k_int = int(k)
                except ValueError:
                    raise HTTPException(status_code=422, detail="k must be an integer")
        if mode == "manual" and k_int is None:
            raise HTTPException(status_code=400, detail="k must be provided in manual mode")
        
        image = await file.read()
        np_img = np.frombuffer(image, np.uint8)
        img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid decoding failed")
        
        result = extract_dominant_colours(
            img_bgr=img_bgr,
            mode=mode,
            k=(k_int if k_int is not None else 5),  # デフォルトは5
        )

        content = jsonable_encoder(
            result,
            custom_encoder={
                np.integer: int,
                np.floating: float,
                np.ndarray: lambda x: x.tolist()
            },
        )
        return JSONResponse(content=content)
    
    except HTTPException:
        raise
    except Exception as e:
        print("EXTRACT FAILED:", repr(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error (see server logs for details)")
 
 
""" 260216, colours_webからコピペ
- --- IGNORE ---
# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from colour import extract_dominant_colours
from typing import Optional

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
    k: Optional[int] = Form(None)
):
    if mode == "manual" and k is None:
        return {"error": "k must be provided in manual mode"}
    

    image = await file.read()
    np_img = np.frombuffer(image, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    result = extract_dominant_colours(
        img_bgr=img_bgr,
        mode=mode,
        k=k or 5
    )

    return result

    # 画像を numpy 配列に変換
    # contents = await file.read()
    # np_img = np.frombuffer(contents, np.uint8)
    # img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # if img is None:
    #     return {"error": "Invalid image file"}

    # result = extract_dominant_colours(
    #     img_bgr=img,
    #     mode=mode,
    #     k=k
    # )

    # return result
"""
