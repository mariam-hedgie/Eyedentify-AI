# OpenCV for reading image files from disk + converting color formats
import cv2
import numpy as np

# tqdm makes progress bar
from tqdm import tqdm


import os

# step 1 - blur detection

def compute_fft_blur_score(image, size=60):
    """
    Computes blur score using high-frequency content removed from FFT spectrum.
    """

    # converts image to grayscale
    # fft works on intensity (brightness)
    # color adds unecessary complexity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gets height and width 
    (h, w) = gray.shape

    # calculates center pixel coordinates
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # computes 2d fast fourier transform of img
    fft = np.fft.fft2(gray)

    # moves low freq components to the center of spectrum
    fftShift = np.fft.fftshift(fft)

    # removes  low freq (blurry info) from the center
    # size 60 means a 120x120 box
    # can be tweaked to see how sensitive sharpness becomes
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0

    # moves freqeuncies back to original quadrants
    fftShift = np.fft.ifftshift(fftShift)

    # inverses fft and now img doesn't have low freqs
    recon = np.fft.ifft2(fftShift)

    # computes log scaled magnitude of img
    magnitude = 20 * np.log(np.abs(recon) + 1e-8)  # avoid log(0)

    # averages high freq content in recon img
    mean_val = np.mean(magnitude)

    return mean_val

# checks if img is blurry based on fft score being < threshold
def check_blurry_fft(img, threshold):
    """
    Returns True if image is blurry based on FFT mean magnitude score.
    """
    score = compute_fft_blur_score(img)
    return score < threshold, score

def filter_sharp_images(input_path, output_path, dynamic_threshold, label):
    """
    Filters out blurry images using FFT sharpness and saves sharp ones.
    """
    os.makedirs(output_path, exist_ok=True)
    retained = []

    for fname in tqdm(os.listdir(input_path), desc=f"Filtering {label}"):
        fpath = os.path.join(input_path, fname)
        img = cv2.imread(fpath)

        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            continue

        is_blurry, score = check_blurry_fft(img, dynamic_threshold)
        if not is_blurry:
            cv2.imwrite(os.path.join(output_path, fname), img)
            retained.append((fname, score))

    return retained

# step 2 - eye splitting

def split_eye_image(image, fname_prefix, save_dir):
    """
    Splits image horizontally if likely to contain two eyes.
    """
    os.makedirs(save_dir, exist_ok=True)
    h, w = image.shape[:2]

    if w / h > 1.5:  # heuristic: wide images likely have both eyes
        mid = w // 2
        left = image[:, :mid]
        right = image[:, mid:]

        cv2.imwrite(os.path.join(save_dir, f"{fname_prefix}_left.jpg"), left)
        cv2.imwrite(os.path.join(save_dir, f"{fname_prefix}_right.jpg"), right)
    else:
        cv2.imwrite(os.path.join(save_dir, f"{fname_prefix}.jpg"), image)

# step 3 - resize to 224
def resize_exact(image, size=224):
    """
    Resizes image exactly to target square size.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

def resize_and_save_all(input_path, output_path, size=224):
    """
    Resizes all images in input_path and saves to output_path.
    """
    os.makedirs(output_path, exist_ok=True)

    for fname in tqdm(os.listdir(input_path), desc="Resizing to 224x224"):
        fpath = os.path.join(input_path, fname)
        img = cv2.imread(fpath)

        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            continue

        resized = resize_exact(img, size)
        cv2.imwrite(os.path.join(output_path, fname), resized)
        
'''
def resize_with_padding(image, target_size=512):
# pad it with black borders to make it square (target_size x target_size).


    # get original height and weight
    h, w = image.shape[:2]

    # calculate scaling factor so longer size becomes target size
    scale = target_size / max(h, w)

    # compute new dimensions to keep aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    
    # resize the img to new dimensions
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # calculate how much padding is needed for 512x512
    delta_w = target_size - new_w # delta width
    delta_h = target_size - new_h # delta height

    # divide padding equally on all sides
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]  # black padding

    # add padding to resized img
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # return final img of 512x512
    return padded_image
'''






'''
def clean_resize_imgs(input_path, output_path, label):

    skipped_files = []

    # lists all filed in input_path dir
    # wraps list in progress bar and adds description label
    for file in tqdm(os.listdir(input_path), desc=f"{label} images"):
        img_path = os.path.join(input_path, file)

        try:
            # loads the image
            img = cv2.imread(img_path)

            # skip corrupt (unreadable or greyscale imgs)
            # cv2.imread returns None if cannot read file
            # if not 3D (height, width, channels) (greyscale would be 2d)
            # checks len is exactly 3 which is RGB/BGR only
            # adds skipped file to list
            if img is None:
                skipped_files.append((file, "Unreadable (cv2.imread returned None)"))
                continue

            if len(img.shape) != 3 or img.shape[2] != 3:
                skipped_files.append((file, "Not a 3-channel RGB image"))
                continue

            # resize to 512x512 with padding
            padded_img = resize_with_padding(img, target_size=512)

            # save to output folder
            # will overwrite existing files with same name without warning
            cv2.imwrite(os.path.join(output_path, file), padded_img)

        except Exception as e:
            print(f"Error processing {file}: {e}")
    return skipped_files
'''