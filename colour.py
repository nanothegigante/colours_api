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
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(data)
        sse.append(km.inertia_)
    kl = KneeLocator(ks, sse, curve="convex", direction="decreasing")
    return int(kl.knee) if kl.knee is not None else int(kmin)
    # if kl.knee is not None:
    #     return int(kl.knee)
    # return int(kmin)


def image_to_base64(img):
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    # _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def _resize_keep_aspect(img_bgr, target_width: int):
    if target_width <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if w <= target_width:
        return img_bgr
    new_w = int(target_width)
    new_h = max(1, int(h * (new_w / w)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def extract_dominant_colours(
    img_bgr,
    mode="auto",
    k=5,
    kmin=2,
    kmax=10,
    resize_width=400,
    include_masks: bool = False, # added in 260219
    mask_resize_width: int = 200, # added in 260224
):
    # resize
    img = _resize_keep_aspect(img_bgr, target_width=resize_width)
    # h, w = img_bgr.shape[:2]
    # new_w = resize_width
    # new_h = int(h * (resize_width / w))
    # img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # クラスタ数決定
    if mode == "auto":
        k = optimal_k(pixels, kmin=kmin, kmax=kmax)

    k = int(k)  # Ensure k is an integer

    # KMeans
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.uint8)

    counts = np.bincount(labels, minlength=k)
    ratios = counts / counts.sum() if counts.sum() > 0 else np.zeros(k, dtype=float)

    labels_img = labels.reshape(img.shape[:2])

    results = []

    for i in range(k):
        hsv_color = centers[i]
        hsv_pixel = np.uint8([[hsv_color]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        hex_code = "#{:02X}{:02X}{:02X}".format(*rgb)

        item = {
            "id": int(i),
            "hex": hex_code,
            "ratio": float(ratios[i]),
        }

        #必要な時だけマスク生成
        if include_masks:
            part = np.ones_like(img) * 255
            mask = labels_img == i
            part[mask] = img[mask]
            part_small = _resize_keep_aspect(part, int(mask_resize_width))
            item["mask_image"] = image_to_base64(part_small)

        results.append(item)

        # # パーティション画像
        # part = np.ones_like(img) * 255
        # mask = labels_img == i
        # part[mask] = img[mask]

        # results.append({
        #     "hex": hex_code,
        #     "ratio": float(ratios[i]),
        #     "mask_image": image_to_base64(part),
        # })

    # 面積順にソート
    results.sort(key=lambda x: x["ratio"], reverse=True)

    return {
        "k": int(k),
        "colours": results
    }

""" 260216, colours_webからコピペ
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
"""