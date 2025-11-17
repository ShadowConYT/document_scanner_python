
from typing import Optional
import numpy as np
import cv2
import math
import itertools
from scipy.spatial import distance as dist
from PIL import Image
from pylsd.lsd import lsd
from pyzbar import pyzbar

MIN_QUAD_AREA_RATIO = 0.25
MAX_QUAD_ANGLE_RANGE = 40
RESCALED_HEIGHT = 1536.0


def resize(image, width: Optional[int] = None, height: Optional[int] = None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None: return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def filter_corners(corners, min_dist=20):
    def predicate(representatives, corner):
        return all(dist.euclidean(representative, corner) >= min_dist for representative in representatives)
    filtered_corners = []
    for c in corners:
        if predicate(filtered_corners, c): filtered_corners.append(c)
    return filtered_corners


def angle_between_vectors_degrees(u, v):
    return np.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

def get_angle(p1, p2, p3):
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))
    avec = a - b
    cvec = c - b
    return angle_between_vectors_degrees(avec, cvec)


def angle_range(quad):
    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])
    angles = [ura, ula, lra, lla]
    return np.ptp(angles)


def get_corners(img):
    lines = lsd(img)
    corners = []
    if lines is not None:
        lines = lines.squeeze().astype(np.int32).tolist()
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2, _ = line
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
            else:
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

        lines = []
        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_x, max_x = np.amin(contour[:, 0], axis=0) + 2, np.amax(contour[:, 0], axis=0) - 2
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            lines.append((min_x, left_y, max_x, right_y))
            cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
            corners.append((min_x, left_y))
            corners.append((max_x, right_y))

        (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y, max_y = np.amin(contour[:, 1], axis=0) + 2, np.amax(contour[:, 1], axis=0) - 2
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            lines.append((top_x, min_y, bottom_x, max_y))
            cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
            corners.append((top_x, min_y))
            corners.append((bottom_x, max_y))

        corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
        corners += zip(corners_x, corners_y)

    corners = filter_corners(corners)
    return corners


def is_valid_contour(cnt, IM_WIDTH, IM_HEIGHT):
    return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * MIN_QUAD_AREA_RATIO and angle_range(cnt) < MAX_QUAD_ANGLE_RANGE)


def get_contour(rescaled_image):
    MORPH = 9
    CANNY = 84
    IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape
    gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(dilated, 0, CANNY)

    test_corners = get_corners(edged)
    approx_contours = []
    if len(test_corners) >= 4:
        quads = []
        for quad in itertools.combinations(test_corners, 4):
            points = np.array(quad)
            points = order_points(points)
            points = np.array([[p] for p in points], dtype="int32")
            quads.append(points)
        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
        quads = sorted(quads, key=angle_range)
        approx = quads[0]
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)

    (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        approx = cv2.approxPolyDP(c, 80, True)
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)
            break

    if not approx_contours:
        TOP_RIGHT, BOTTOM_RIGHT = (IM_WIDTH, 0), (IM_WIDTH, IM_HEIGHT)
        BOTTOM_LEFT, TOP_LEFT = (0, IM_HEIGHT), (0, 0)
        screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
    else:
        screenCnt = max(approx_contours, key=cv2.contourArea)

    return screenCnt.reshape(4, 2)
# ---------------------------------------------------------------------------------------------------------------------


def process_image(image: np.ndarray) -> np.ndarray:
    ratio = image.shape[0] / RESCALED_HEIGHT
    original_image = image.copy()
    rescaled_image = resize(image, height=int(RESCALED_HEIGHT))
    screenCnt = get_contour(rescaled_image)

    rescaled_with_contour = rescaled_image.copy()
    cv2.drawContours(rescaled_with_contour, [screenCnt.astype(int)], -1, (0, 255, 0), 3)
    warped = four_point_transform(original_image, screenCnt * ratio)
    return warped

# ---------------------------------------------------------------------------------------------------------------------


def find_and_decode_barcodes(img: np.ndarray):
    img_with_boxes = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_barcodes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        aspect_ratio = w / float(h)
        if w > 150 and aspect_ratio > 2.5:
            padding = 15
            roi_x1 = max(0, x - padding)
            roi_y1 = max(0, y - padding)
            roi_x2 = min(img.shape[1], x + w + padding)
            roi_y2 = min(img.shape[0], y + h + padding)

            roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            decoded_text = None

            results_pyzbar = pyzbar.decode(roi_gray)
            if results_pyzbar:
                decoded_text = results_pyzbar[0].data.decode("utf-8")
                print(f"Decoded '{decoded_text}' from region at x={x}, y={y}")

            if decoded_text:
                found_barcodes.append({"text": decoded_text, "box_2d": [x, y, x + w, y + h]})
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img_with_boxes, decoded_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("\n--- Final Results ---")
    if not found_barcodes: print("No barcodes were successfully decoded.")
    else:
        for bc in found_barcodes: print(bc)

    return found_barcodes


def get_top_right_barcode_text(barcodes: list[dict]) -> Optional[str]:
    if not barcodes:
        return None
    sorted_barcodes = sorted(
        barcodes, key=lambda b: (-b['box_2d'][2], b['box_2d'][1]))
    return sorted_barcodes[0]['text']

# ---------------------------------------------------------------------------------------------------------------------


def get_filename(first_page_img: np.ndarray) -> Optional[str]:
    all_barcodes = find_and_decode_barcodes(first_page_img)
    return get_top_right_barcode_text(all_barcodes)

# ---------------------------------------------------------------------------------------------------------------------


def save_as_pdf(pages: list[np.ndarray], filename: str):
    if not pages:
        raise ValueError("Input image list cannot be empty.")
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    A4_WIDTH, A4_HEIGHT = 595*2, 842*2
    pdf_pages = []

    for image_array in pages:
        pil_image = Image.fromarray(image_array)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        img_w, img_h = pil_image.size

        scale_w = A4_WIDTH / img_w
        scale_h = A4_HEIGHT / img_h
        scale = min(scale_w, scale_h)

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized_image = pil_image.resize(
            (new_w, new_h), Image.Resampling.LANCZOS)

        a4_page = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), (255, 255, 255))

        paste_x = (A4_WIDTH - new_w) // 2
        paste_y = (A4_HEIGHT - new_h) // 2

        a4_page.paste(resized_image, (paste_x, paste_y))
        pdf_pages.append(a4_page)

    if pdf_pages:
        pdf_pages[0].save(filename, save_all=True, append_images=pdf_pages[1:])
