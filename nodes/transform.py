import numpy as np
import cv2
import torch

class Megatran:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "canvas": ("IMAGE",),
            },
            "required": {
                "position_x": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                "position_y": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                "offset_x": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "offset_y": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "pivot_x": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                "pivot_y": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "canvas_width": ("INT", {"default": 1024, "min": 1, "max": 10000}),
                "canvas_height": ("INT", {"default": 1024, "min": 1, "max": 10000}),
                "expand_canvas": ("BOOLEAN", {"default": False}),
                "show_pivot": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "position_x", "position_y", "offset_x", "offset_y")
    CATEGORY = "CS Custom Nodes/Transform"
    FUNCTION = "execute"

    def execute(
        self, position_x, position_y, offset_x, offset_y, pivot_x, pivot_y,
        rotation, scale, canvas_width, canvas_height, expand_canvas, show_pivot,
        image=None, mask=None, canvas=None
    ):
        # Always sum position and offset, regardless of slider or input
        tx = position_x + offset_x
        ty = position_y + offset_y

        transformed_image = None
        transformed_mask = None

        # Determine canvas dimensions
        if canvas is not None:
            height, width = self.to_numpy(canvas).shape[:2]
            canvas_width, canvas_height = width, height

        # Apply transformations
        if image is not None:
            transformed_image = self.apply_transform(
                self.to_numpy(image), tx, ty, pivot_x, pivot_y, rotation, scale,
                canvas_width, canvas_height, expand_canvas, show_pivot
            )
            transformed_image = self.to_tensor(transformed_image)

        if mask is not None:
            transformed_mask = self.apply_transform(
                self.to_numpy(mask), tx, ty, pivot_x, pivot_y, rotation, scale,
                canvas_width, canvas_height, expand_canvas, show_pivot
            )
            transformed_mask = self.to_tensor(transformed_mask)

        # Output the current values for the sliders/inputs
        return transformed_image, transformed_mask, position_x, position_y, offset_x, offset_y

    def apply_transform(
        self, img, x, y, pivot_x, pivot_y, rotation, scale,
        canvas_width, canvas_height, expand_canvas, show_pivot
    ):
        rows, cols = img.shape[:2]

        # If expanding canvas, calculate the bounding box for the transformed image
        if expand_canvas:
            corners = np.array([
                [0, 0],
                [cols, 0],
                [cols, rows],
                [0, rows]
            ])
            center = np.array([pivot_x, pivot_y])
            rotation_matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), rotation, scale)
            rotated_corners = cv2.transform(np.array([corners - center]), rotation_matrix[:, :2])[0] + center
            x_coords, y_coords = rotated_corners[:, 0], rotated_corners[:, 1]
            new_width = int(max(x_coords) - min(x_coords))
            new_height = int(max(y_coords) - min(y_coords))
            canvas_width = max(canvas_width, new_width + abs(x))
            canvas_height = max(canvas_height, new_height + abs(y))

            if img.ndim == 2:
                canvas = np.zeros((canvas_height, canvas_width), dtype=img.dtype)
            else:
                canvas = np.zeros((canvas_height, canvas_width, img.shape[2]), dtype=img.dtype)

            center_x, center_y = canvas_width // 2, canvas_height // 2
            top_left_x = center_x - cols // 2
            top_left_y = center_y - rows // 2
            canvas[top_left_y:top_left_y+rows, top_left_x:top_left_x+cols] = img
            img = canvas
            pivot_x += top_left_x
            pivot_y += top_left_y

        # Scaling matrix around the pivot point
        M_scale = cv2.getRotationMatrix2D((pivot_x, pivot_y), 0, scale)
        img_scaled = cv2.warpAffine(img, M_scale, (canvas_width, canvas_height))

        # Rotation matrix around the pivot point
        M_rotate = cv2.getRotationMatrix2D((pivot_x, pivot_y), rotation, 1)
        img_rotated = cv2.warpAffine(img_scaled, M_rotate, (canvas_width, canvas_height))

        # Translation matrix
        M_translate = np.float32([[1, 0, x], [0, 1, y]])
        img_transformed = cv2.warpAffine(img_rotated, M_translate, (canvas_width, canvas_height))

        # Add a red dot to indicate the pivot point after all transformations
        if show_pivot:
            final_pivot_x = int(pivot_x + x)
            final_pivot_y = int(pivot_y + y)
            img_transformed = cv2.circle(img_transformed, (final_pivot_x, final_pivot_y), radius=5, color=(0, 0, 255), thickness=-1)

        return img_transformed

    def to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.squeeze(img)
        return img

    def to_tensor(self, img):
        img_tensor = torch.tensor(img)
        if img_tensor.ndimension() == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
