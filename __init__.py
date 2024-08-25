from .nodes import ImageStitchingNode


NODE_CLASS_MAPPINGS = {
    "Image Stitching Node": ImageStitchingNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Stitching Node": "Panoramic Image Stitcher",
}


print("\033[34mComfyUI Custom Nodes: \033[92mLoaded Image Stitching Node\033[0m")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']