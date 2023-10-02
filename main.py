import cv2
import numpy as np


def load_and_convert_to_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def calculate_entropy(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram /= histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))

    return entropy

def discretize_image(image, step_size):
    discretized_image = image[::step_size, ::step_size]
    return discretized_image

def quantize_image(image, levels):
    quantized_image = (np.floor(image / (256 / levels)) * (256 / levels)).astype(np.uint8)
    return quantized_image

def reconstruct_image(quantized_image, original_image, interpolation):
    reconstructed_image = cv2.resize(quantized_image, original_image.shape[:2][::-1], interpolation=interpolation)
    return reconstructed_image


def print_entropy_and_image(entropy_value, reconstructed_image, title):
    entropy_str = f"{entropy_value:.4f}"
    print(f"Ентропія {title}: {entropy_str}")
    cv2.imshow(title, reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#First task
def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def display_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#First task


def main(image_path, quantization_levels):
    # First task
    image = load_image(image_path)
    display_image(image, "Original Image")

    grayscale_image = convert_to_grayscale(image)
    display_image(grayscale_image, "Grayscale Image")
    # First task

    # Second task
    entropy_first = calculate_entropy(image)
    print(f"Ентропія початкового зображення: {entropy_first:.4f}")
    # Second task

    # Third task
    step_size_2 = 2
    discretized_image_2 = discretize_image(image, step_size_2)
    cv2.imwrite('discretized_image_2.jpg', discretized_image_2)
    entropy_discretized_image_2 = calculate_entropy(discretized_image_2)
    print(f"Ентропія дискретизованого на 2 зображення: {entropy_discretized_image_2:.4f}")

    step_size_4 = 4
    discretized_image_4 = discretize_image(image, step_size_4)
    cv2.imwrite('discretized_image_4.jpg', discretized_image_4)
    entropy_discretized_image_4 = calculate_entropy(discretized_image_4)
    print(f"Ентропія дискретизованого на 4 зображення: {entropy_discretized_image_4:.4f}")
    # Third task

    # Fourth, Five task
    images = [image, discretized_image_2, discretized_image_4]

    for img in images:
        print("==============================================")
        print(f"Ентропія зображення перед квантуванням:")
        entropy_value = calculate_entropy(img)
        print(f"{entropy_value:.4f}")

        for levels in quantization_levels:
            quantized_image = quantize_image(img, levels)
            print(f"Ентропія зображення після квантування на {levels} рівнів:")
            entropy_after_quantization = calculate_entropy(quantized_image)
            print(f"{entropy_after_quantization:.4f}")
            display_image(quantized_image, f"Quantized Image (Levels {levels})")
    # Fourth, Five task

    print("/////////////////////")

    print("Ентропія відновлених зображень після дискретизації ")

    # Sixth, Seventh task
    reconstructed_image_nearest_2 = reconstruct_image(discretized_image_2, image, cv2.INTER_NEAREST)
    reconstructed_image_linear_2 = reconstruct_image(discretized_image_2, image, cv2.INTER_LINEAR)
    reconstructed_image_cubic_2 = reconstruct_image(discretized_image_2, image, cv2.INTER_CUBIC)

    entropy_reconstructed_image_nearest_2 = calculate_entropy(reconstructed_image_nearest_2)
    entropy_reconstructed_image_linear_2 = calculate_entropy(reconstructed_image_linear_2)
    entropy_reconstructed_image_cubic_2 = calculate_entropy(reconstructed_image_cubic_2)

    reconstructed_image_nearest_4 = reconstruct_image(discretized_image_4, image, cv2.INTER_NEAREST)
    reconstructed_image_linear_4 = reconstruct_image(discretized_image_4, image, cv2.INTER_LINEAR)
    reconstructed_image_cubic_4 = reconstruct_image(discretized_image_4, image, cv2.INTER_CUBIC)

    entropy_reconstructed_image_nearest_4 = calculate_entropy(reconstructed_image_nearest_4)
    entropy_reconstructed_image_linear_4 = calculate_entropy(reconstructed_image_linear_4)
    entropy_reconstructed_image_cubic_4 = calculate_entropy(reconstructed_image_cubic_4)


    print_entropy_and_image(entropy_reconstructed_image_nearest_2, reconstructed_image_nearest_2, "Entropy Reconstructed (Nearest, 2)")
    print_entropy_and_image(entropy_reconstructed_image_linear_2, reconstructed_image_linear_2, "Entropy Reconstructed (Linear, 2)")
    print_entropy_and_image(entropy_reconstructed_image_cubic_2, reconstructed_image_cubic_2, "Entropy Reconstructed (Cubic, 2)")

    print_entropy_and_image(entropy_reconstructed_image_nearest_4, reconstructed_image_nearest_4, "Entropy Reconstructed (Nearest, 4)")
    print_entropy_and_image(entropy_reconstructed_image_linear_4, reconstructed_image_linear_4, "Entropy Reconstructed (Linear, 4)")
    print_entropy_and_image(entropy_reconstructed_image_cubic_4, reconstructed_image_cubic_4, "Entropy Reconstructed (Cubic, 4)")
    # Sixth, Seventh task



if __name__ == "__main__":
    image_path = 'image.jpg'  #your path to image
    quantization_levels = [8, 16, 64]
    main(image_path, quantization_levels)
