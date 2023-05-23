from semantic_extractor import SemanticBased

if __name__ == "__main__":
    # Semantic Extraction for masks and foreground
    semantic_images = SemanticBased(image_name="bikers.png")
    semantic_images.extract_images(color_image=True)