
import os
import argparse
import cv2
import numpy as np
import sys

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_img(path, img):
    success = cv2.imwrite(path, img)
    if not success:
        print(f"Failed to save: {path}")
    else:
        print(f"Saved: {path}")

def process_image(input_path, out_folder, show=False):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_folder = os.path.dirname(input_path) or os.getcwd()
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    out_folder = out_folder if out_folder else os.path.join(base_folder, "converted")
    ensure_dir(out_folder)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image (maybe unsupported format): {input_path}")

    b, g, r = cv2.split(img)
    height, width = b.shape
    zeros = np.zeros((height, width), dtype=b.dtype)

    blue_img = cv2.merge([b, zeros, zeros])
    save_img(os.path.join(out_folder, f"{base_name}_blue.png"), blue_img)

    green_img = cv2.merge([zeros, g, zeros])
    save_img(os.path.join(out_folder, f"{base_name}_green.png"), green_img)

    red_img = cv2.merge([zeros, zeros, r])
    save_img(os.path.join(out_folder, f"{base_name}_red.png"), red_img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    save_img(os.path.join(out_folder, f"{base_name}_hsv_bgr.png"), hsv_bgr)

    h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h_vis = cv2.merge([h_norm, h_norm, h_norm])
    save_img(os.path.join(out_folder, f"{base_name}_h_channel.png"), h_vis)

    s_vis = cv2.merge([s, s, s])
    save_img(os.path.join(out_folder, f"{base_name}_s_channel.png"), s_vis)

    v_vis = cv2.merge([v, v, v])
    save_img(os.path.join(out_folder, f"{base_name}_v_channel.png"), v_vis)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_img(os.path.join(out_folder, f"{base_name}_grayscale.png"), gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_img(os.path.join(out_folder, f"{base_name}_binary_otsu.png"), binary)

    print("\nDone. All images saved to:", out_folder)

    if show:
        display_list = [
            ("original", img),
            ("red", red_img),
            ("green", green_img),
            ("blue", blue_img),
            ("hsv_bgr", hsv_bgr),
            ("h_channel", h_vis),
            ("s_channel", s_vis),
            ("v_channel", v_vis),
            ("grayscale", gray),
            ("binary", binary),
        ]
        for (title, im) in display_list:
            cv2.imshow(title, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    default_input = r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\bakugan.jpg"
    parser = argparse.ArgumentParser(description="Convert image into several color/channel variants.")
    parser.add_argument("--input", "-i", default=default_input, help=f"Path to input image (default: {default_input})")
    parser.add_argument("--output", "-o", default=None, help="Output folder (default: <input_folder>/converted)")
    parser.add_argument("--show", action="store_true", help="Show images using OpenCV windows (optional).")
    args = parser.parse_args()

    try:
        process_image(args.input, args.output, args.show)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
