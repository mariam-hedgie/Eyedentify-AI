# OpenCV for reading image files from disk + converting color formats
import cv2
import numpy as np

# tqdm makes progress bar
from tqdm import tqdm


import os

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

            # resize to standard size
            # can be changed later to optimize accuracy vs speed vs memory use
            try:
                resized_img = cv2.resize(img, (224, 224))
            except Exception as e:
                skipped_files.append((file, f"Resize error: {str(e)}"))
                continue

            # save to output folder
            # will overwrite existing files with same name without warning
            cv2.imwrite(os.path.join(output_path, file), resized_img)

        except Exception as e:
            print(f"Error processing {file}: {e}")
    return skipped_files

def compute_fft_blur_score(image, size=60):

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
    score = compute_fft_blur_score(img)
    return score < threshold, score

