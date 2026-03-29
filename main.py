import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_impulse_noise(image, prob=0.20):
   
    noisy = np.copy(image)
    h, w = image.shape[:2]

   
    mask = np.random.rand(h, w) < prob

   
   
    for c in range(image.shape[2]):
        noisy[mask, c] = np.random.randint(0, 256, size=np.sum(mask))
 

    return noisy.astype(np.uint8)


def add_even_noise(image, scale=100.0):
    
    noise = np.random.uniform(-scale , scale , image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def filter_via_average(image, ksize=5):
 
    return cv2.blur(image, (ksize, ksize))


def filter_via_median(image, ksize=5):
  
    return cv2.medianBlur(image, ksize)


def filter_via_gauss(image, ksize=5, sigma=1.0):

    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def load_image(path):
   
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def exhibit_changes(image1, image2, title1="Image 1", title2="Image 2"):

    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

  
    img1_impulse = add_impulse_noise(image1)
    img1_even = add_even_noise(image1)

    img2_impulse = add_impulse_noise(image2)
    img2_even = add_even_noise(image2)

    filters = [
        ("Average", filter_via_average),
        ("Median", filter_via_median),
        ("Gauss", filter_via_gauss),
    ]

    fig, axes = plt.subplots(4, 6, figsize=(16, 12))

  
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title(title1)
    axes[0, 1].imshow(cv2.cvtColor(img1_impulse, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Impulse")
    axes[0, 2].imshow(cv2.cvtColor(img1_even, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Even (uniform)")

    axes[0, 3].imshow(img2_rgb)
    axes[0, 3].set_title(title2)
    axes[0, 4].imshow(cv2.cvtColor(img2_impulse, cv2.COLOR_BGR2RGB))
    axes[0, 4].set_title("Impulse")
    axes[0, 5].imshow(cv2.cvtColor(img2_even, cv2.COLOR_BGR2RGB))
    axes[0, 5].set_title("Even")

    for row_idx, (name, filt) in enumerate(filters, 1):
        
        img1_impulse_filt = filt(img1_impulse)
        img2_impulse_filt = filt(img2_impulse)

        axes[row_idx, 0].imshow(img1_rgb)
        axes[row_idx, 0].set_title(f"{title1} (orig)")
        axes[row_idx, 1].imshow(cv2.cvtColor(img1_impulse_filt, cv2.COLOR_BGR2RGB))
        axes[row_idx, 1].set_title(f"Impulse → {name}")

        axes[row_idx, 3].imshow(img2_rgb)
        axes[row_idx, 3].set_title(f"{title2} (orig)")
        axes[row_idx, 4].imshow(cv2.cvtColor(img2_impulse_filt, cv2.COLOR_BGR2RGB))
        axes[row_idx, 4].set_title(f"Impulse → {name}")

    
        img1_even_filt = filt(img1_even)
        img2_even_filt = filt(img2_even)

        axes[row_idx, 2].imshow(cv2.cvtColor(img1_even_filt, cv2.COLOR_BGR2RGB))
        axes[row_idx, 2].set_title(f"Even → {name}")

        axes[row_idx, 5].imshow(cv2.cvtColor(img2_even_filt, cv2.COLOR_BGR2RGB))
        axes[row_idx, 5].set_title(f"Even → {name}")

    for ax in axes.flat:
        ax.axis("off")
   
    
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    plt.close()



def main():
  
    path1 = "image1.jpg"
    path2 = "gray.jpg"

    img1 = load_image(path1)
    img2 = load_image(path2)

    exhibit_changes(img1, img2, title1="Image 1", title2="Image 2")


if __name__ == "__main__":
    main()