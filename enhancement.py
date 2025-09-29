# enhancement.py (updated: stronger/more visible histogram waves)
# Requirements: pip install opencv-python matplotlib numpy
# Run example:
# python "C:\Users\bumii\Desktop\GRAYSCALE IMAGE\enhancement.py" --hist_vis_mode scale --hist_vis_scale 8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def compute_histogram(arr):
    counts, bins = np.histogram(arr.flatten(), bins=256, range=(0,256))
    return counts.astype(np.float64), bins

def transform_counts_for_display(counts, mode='auto', scale=1.0):
    """
    mode:
      - 'auto'      : raw counts (as before)
      - 'normalize' : scale so max -> 100 (percent)
      - 'scale'     : multiply raw counts by given scale factor
      - 'log'       : log1p then normalize to 100 for display
    returns: transformed counts (float) and suggested ymax
    """
    if mode == 'normalize':
        mx = counts.max()
        if mx <= 0:
            disp = counts
        else:
            disp = (counts / mx) * 100.0
        ymax = max(1.0, disp.max() * 1.05)
        return disp, ymax

    if mode == 'scale':
        disp = counts * float(scale)
        ymax = max(1.0, disp.max() * 1.06)
        return disp, ymax

    if mode == 'log':
        # log1p reduces dynamic range; normalize to 100 to be visually big
        disp = np.log1p(counts)
        mx = disp.max()
        if mx <= 0:
            norm = disp
        else:
            norm = (disp / mx) * 100.0
        ymax = max(1.0, norm.max() * 1.05)
        return norm, ymax

    # default 'auto'
    ymax = max(1.0, counts.max() * 1.05)
    return counts, ymax

# Generic plotting helpers (larger hist area, stronger visuals)
def plot_image_and_hist_grayscale(orig_gray, adjusted_gray, title_prefix, out_dir, hist_mode='auto', hist_scale=1.0):
    counts_orig, bins = compute_histogram(orig_gray)
    counts_adj, _ = compute_histogram(adjusted_gray)

    disp_orig, ymax_orig = transform_counts_for_display(counts_orig, mode=hist_mode, scale=hist_scale)
    disp_adj, ymax_adj = transform_counts_for_display(counts_adj, mode=hist_mode, scale=hist_scale)
    ymax = max(ymax_orig, ymax_adj)

    fig, axes = plt.subplots(2, 2, figsize=(14,10), gridspec_kw={'height_ratios':[1,1.7]})
    fig.suptitle(title_prefix, fontsize=18)

    # images
    axes[0,0].imshow(orig_gray, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title('Original (grayscale)', fontsize=14)
    axes[0,0].axis('off')

    axes[0,1].imshow(adjusted_gray, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title('Adjusted (grayscale)', fontsize=14)
    axes[0,1].axis('off')

    # histograms as bars + filled area + bold line
    w = 1.0
    axes[1,0].bar(bins[:-1], disp_orig, width=w, alpha=0.45, align='edge', edgecolor='k')
    axes[1,0].fill_between(bins[:-1], disp_orig, step='mid', alpha=0.25)
    axes[1,0].plot(bins[:-1], disp_orig, linewidth=2.0)
    axes[1,0].set_title('Histogram - original', fontsize=13)
    axes[1,0].set_xlim(0,255)
    axes[1,0].set_ylim(0, ymax)
    axes[1,0].tick_params(axis='both', which='major', labelsize=11)

    axes[1,1].bar(bins[:-1], disp_adj, width=w, alpha=0.45, align='edge', edgecolor='k')
    axes[1,1].fill_between(bins[:-1], disp_adj, step='mid', alpha=0.25)
    axes[1,1].plot(bins[:-1], disp_adj, linewidth=2.0)
    axes[1,1].set_title('Histogram - adjusted', fontsize=13)
    axes[1,1].set_xlim(0,255)
    axes[1,1].set_ylim(0, ymax)
    axes[1,1].tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout(rect=[0,0,1,0.96])
    out_path = os.path.join(out_dir, f'{title_prefix.replace(" ","_")}_grayscale_result_vis.png')
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.show()

def plot_channel_image_and_hist(channel_orig, channel_adj, channel_name, out_dir, hist_mode='auto', hist_scale=1.0):
    counts_orig, bins = compute_histogram(channel_orig)
    counts_adj, _ = compute_histogram(channel_adj)

    disp_orig, ymax_orig = transform_counts_for_display(counts_orig, mode=hist_mode, scale=hist_scale)
    disp_adj, ymax_adj = transform_counts_for_display(counts_adj, mode=hist_mode, scale=hist_scale)
    ymax = max(ymax_orig, ymax_adj)

    fig, axes = plt.subplots(2, 2, figsize=(12,10), gridspec_kw={'height_ratios':[1,1.8]})
    fig.suptitle(f'Channel {channel_name}', fontsize=16)

    axes[0,0].imshow(channel_orig, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title(f'Original {channel_name}', fontsize=13)
    axes[0,0].axis('off')

    axes[0,1].imshow(channel_adj, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title(f'Adjusted {channel_name}', fontsize=13)
    axes[0,1].axis('off')

    axes[1,0].bar(bins[:-1], disp_orig, width=1.0, alpha=0.45, align='edge', edgecolor='k')
    axes[1,0].plot(bins[:-1], disp_orig, linewidth=2.0)
    axes[1,0].set_title('Histogram - original', fontsize=12)
    axes[1,0].set_xlim(0,255)
    axes[1,0].set_ylim(0, ymax)
    axes[1,0].tick_params(axis='both', which='major', labelsize=10)

    axes[1,1].bar(bins[:-1], disp_adj, width=1.0, alpha=0.45, align='edge', edgecolor='k')
    axes[1,1].plot(bins[:-1], disp_adj, linewidth=2.0)
    axes[1,1].set_title('Histogram - adjusted', fontsize=12)
    axes[1,1].set_xlim(0,255)
    axes[1,1].set_ylim(0, ymax)
    axes[1,1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = os.path.join(out_dir, f'channel_{channel_name}_result_vis.png')
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.show()

def plot_color_image_and_channel_hists(orig_bgr, adjusted_bgr, out_dir, hist_mode='auto', hist_scale=1.0):
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    adj_rgb = cv2.cvtColor(adjusted_bgr, cv2.COLOR_BGR2RGB)

    counts_orig = [compute_histogram(orig_bgr[:,:,i])[0] for i in range(3)]
    counts_adj = [compute_histogram(adjusted_bgr[:,:,i])[0] for i in range(3)]

    disp = [transform_counts_for_display(counts_orig[i], mode=hist_mode, scale=hist_scale)[0] for i in range(3)]
    disp_adj = [transform_counts_for_display(counts_adj[i], mode=hist_mode, scale=hist_scale)[0] for i in range(3)]
    ymaxs = [transform_counts_for_display(counts_orig[i], mode=hist_mode, scale=hist_scale)[1] for i in range(3)]
    ymaxs_adj = [transform_counts_for_display(counts_adj[i], mode=hist_mode, scale=hist_scale)[1] for i in range(3)]
    ymaxs_all = [max(ymaxs[i], ymaxs_adj[i]) for i in range(3)]

    fig = plt.figure(figsize=(16,11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1,1.8])

    ax_img1 = fig.add_subplot(gs[0, 0:2])
    ax_img2 = fig.add_subplot(gs[0, 2])

    ax_img1.imshow(orig_rgb)
    ax_img1.set_title('Original (color)', fontsize=14)
    ax_img1.axis('off')

    ax_img2.imshow(adj_rgb)
    ax_img2.set_title('Adjusted merged (color)', fontsize=14)
    ax_img2.axis('off')

    channel_names = ['B','G','R']
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.bar(range(256), disp[i], width=1.0, alpha=0.35, align='edge', edgecolor='k')
        ax.plot(range(256), disp[i], linewidth=1.6)
        ax.plot(range(256), disp_adj[i], linestyle='--', linewidth=1.6)
        ax.set_title(f'Channel {channel_names[i]} histogram', fontsize=12)
        ax.set_xlim(0,255)
        ax.set_ylim(0, ymaxs_all[i])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(['orig','adj'], fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'color_merge_and_hists_vis.png')
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.show()

def ensure_out_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description='Image enhancement tasks for Goku image (grayscale & per-channel).')
    parser.add_argument('--img', type=str,
                        default=r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\goku.jpg",
                        help='Path to goku.jpg (default example path)')
    parser.add_argument('--gray_alpha', type=float, default=1.2, help='Contrast for grayscale (alpha)')
    parser.add_argument('--gray_beta', type=int, default=15, help='Brightness for grayscale (beta)')
    parser.add_argument('--b_alpha', type=float, default=1.1, help='B channel contrast')
    parser.add_argument('--b_beta', type=int, default=5, help='B channel brightness')
    parser.add_argument('--g_alpha', type=float, default=1.25, help='G channel contrast')
    parser.add_argument('--g_beta', type=int, default=10, help='G channel brightness')
    parser.add_argument('--r_alpha', type=float, default=1.05, help='R channel contrast')
    parser.add_argument('--r_beta', type=int, default=20, help='R channel brightness')

    # NEW: histogram visualization options
    parser.add_argument('--hist_vis_mode', type=str, default='auto',
                        choices=['auto','normalize','scale','log'],
                        help="Histogram visual mode: 'auto','normalize','scale','log'")
    parser.add_argument('--hist_vis_scale', type=float, default=1.0,
                        help='When hist_vis_mode==scale, multiply counts by this factor')

    args = parser.parse_args()

    img_path = args.img
    out_dir = os.path.dirname(img_path) or '.'
    ensure_out_dir(out_dir)

    if not os.path.isfile(img_path):
        print(f'ERROR: file not found: {img_path}')
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f'ERROR: OpenCV gagal membaca file (pastikan path dan ekstensi benar): {img_path}')
        return

    print('--- NO.1: Grayscale enhancement ---')
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_adj = adjust_brightness_contrast(gray_orig, alpha=args.gray_alpha, beta=args.gray_beta)
    cv2.imwrite(os.path.join(out_dir, 'goku_gray_original.png'), gray_orig)
    cv2.imwrite(os.path.join(out_dir, 'goku_gray_adjusted.png'), gray_adj)
    plot_image_and_hist_grayscale(gray_orig, gray_adj, 'No1_Grayscale_Enhancement', out_dir,
                                  hist_mode=args.hist_vis_mode, hist_scale=args.hist_vis_scale)

    print('--- NO.2: Color channels (split, adjust, merge) ---')
    B_orig, G_orig, R_orig = cv2.split(img_bgr)
    B_adj = adjust_brightness_contrast(B_orig, alpha=args.b_alpha, beta=args.b_beta)
    G_adj = adjust_brightness_contrast(G_orig, alpha=args.g_alpha, beta=args.g_beta)
    R_adj = adjust_brightness_contrast(R_orig, alpha=args.r_alpha, beta=args.r_beta)

    cv2.imwrite(os.path.join(out_dir, 'goku_B_original.png'), B_orig)
    cv2.imwrite(os.path.join(out_dir, 'goku_G_original.png'), G_orig)
    cv2.imwrite(os.path.join(out_dir, 'goku_R_original.png'), R_orig)
    cv2.imwrite(os.path.join(out_dir, 'goku_B_adjusted.png'), B_adj)
    cv2.imwrite(os.path.join(out_dir, 'goku_G_adjusted.png'), G_adj)
    cv2.imwrite(os.path.join(out_dir, 'goku_R_adjusted.png'), R_adj)

    plot_channel_image_and_hist(B_orig, B_adj, 'B', out_dir,
                                hist_mode=args.hist_vis_mode, hist_scale=args.hist_vis_scale)
    plot_channel_image_and_hist(G_orig, G_adj, 'G', out_dir,
                                hist_mode=args.hist_vis_mode, hist_scale=args.hist_vis_scale)
    plot_channel_image_and_hist(R_orig, R_adj, 'R', out_dir,
                                hist_mode=args.hist_vis_mode, hist_scale=args.hist_vis_scale)

    merged_bgr = cv2.merge([B_adj, G_adj, R_adj])
    merged_path = os.path.join(out_dir, 'goku_merged_adjusted.png')
    cv2.imwrite(merged_path, merged_bgr)
    print(f'Saved merged adjusted color image: {merged_path}')

    plot_color_image_and_channel_hists(img_bgr, merged_bgr, out_dir,
                                      hist_mode=args.hist_vis_mode, hist_scale=args.hist_vis_scale)

    print('--- selesai ---')

if __name__ == '__main__':
    main()
