# colour.py
import cv2
import numpy as np
import base64
from sklearn.cluster import KMeans
from kneed import KneeLocator

def optimal_k(data, kmin=2, kmax=10):
    sse = []
    ks = list(range(kmin, kmax + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        km.fit(data)
        sse.append(km.inertia_)

    kl = KneeLocator(ks, sse, curve="convex", direction="decreasing")
    return kl.knee if kl.knee else kmin


def image_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def extract_dominant_colours(
    img_bgr,
    mode="auto",
    k=5,
    kmin=2,
    kmax=10,
    resize_width=400,
):
    # resize
    h, w = img_bgr.shape[:2]
    new_w = resize_width
    new_h = int(h * (resize_width / w))
    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # クラスタ数決定
    if mode == "auto":
        k = optimal_k(pixels, kmin=kmin, kmax=kmax)

    # KMeans
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.uint8)

    counts = np.bincount(labels, minlength=k)
    ratios = counts / counts.sum()

    labels_img = labels.reshape(img.shape[:2])

    results = []

    for i in range(k):
        hsv_color = centers[i]
        hsv_pixel = np.uint8([[hsv_color]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        hex_code = "#{:02X}{:02X}{:02X}".format(*rgb)

        # パーティション画像
        part = np.ones_like(img) * 255
        mask = labels_img == i
        part[mask] = img[mask]

        results.append({
            "hex": hex_code,
            "ratio": float(ratios[i]),
            "mask_image": image_to_base64(part),
        })

    # 面積順にソート
    results.sort(key=lambda x: x["ratio"], reverse=True)

    return {
        "k": k,
        "colours": results
    }
