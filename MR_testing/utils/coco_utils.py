# coco_reader.py
import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import numpy as np

class COCOAnnotationReader:
    """Handles reading and processing COCO annotations."""
    
    def __init__(self, coco_dir: Union[str, Path]):
        self.coco_dir = Path(coco_dir)
        self.train_annotations = self._load_annotations("train2024")
        self.val_annotations = self._load_annotations("val2024")
        
    def _load_annotations(self, split: str) -> Dict:
        """Load COCO format annotations."""
        ann_file = self.coco_dir / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        with open(ann_file, 'r') as f:
            return json.load(f)
            
    def get_image_info(self, image_id: int, split: str = "train2024") -> Optional[Dict]:
        """Get image information from COCO annotations."""
        annotations = self.train_annotations if "train" in split else self.val_annotations
        for img in annotations['images']:
            if img['id'] == image_id:
                return {
                    **img,
                    'image_path': self.coco_dir / "images" / split / img['file_name']
                }
        return None
        
    def get_annotation(self, ann_id: int, split: str = "train2024") -> Optional[Dict]:
        """Get specific annotation by ID."""
        annotations = self.train_annotations if "train" in split else self.val_annotations
        for ann in annotations['annotations']:
            if ann['id'] == ann_id:
                return ann
        return None

    def get_image_annotations(self, image_id: int, split: str = "train2024") -> List[Dict]:
        """Get all annotations for a specific image."""
        annotations = self.train_annotations if "train" in split else self.val_annotations
        return [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

    def get_points_from_annotation(self, annotation: Dict) -> List[List[float]]:
        """Extract point coordinates from annotation."""
        points = []
        
        # Center point from bbox
        bbox = annotation['bbox']  # [x, y, width, height]
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        points.append([center_x, center_y])
        
        # Points from segmentation if available
        if 'segmentation' in annotation:
            seg = annotation['segmentation']
            if isinstance(seg, list) and len(seg) > 0:
                # Get keypoints from segmentation polygon
                polygon = np.array(seg[0]).reshape(-1, 2)
                # Add top, bottom, left, right extreme points
                top_point = polygon[np.argmin(polygon[:, 1])]
                bottom_point = polygon[np.argmax(polygon[:, 1])]
                left_point = polygon[np.argmin(polygon[:, 0])]
                right_point = polygon[np.argmax(polygon[:, 0])]
                points.extend([top_point.tolist(), bottom_point.tolist(),
                             left_point.tolist(), right_point.tolist()])
                
        return points

    def get_category_info(self, category_id: int, split: str = "train2024") -> Optional[Dict]:
        """Get category information."""
        annotations = self.train_annotations if "train" in split else self.val_annotations
        for cat in annotations['categories']:
            if cat['id'] == category_id:
                return cat
        return None

    def find_suitable_images(self, 
                           category_names: List[str] = None,
                           min_obj_size: float = None,
                           max_instances: int = None) -> List[Dict]:
        """Find images matching specific criteria."""
        suitable_images = []
        
        # Get category IDs if names provided
        category_ids = None
        if category_names:
            category_ids = []
            for cat in self.train_annotations['categories']:
                if cat['name'] in category_names:
                    category_ids.append(cat['id'])
        
        # Search through annotations
        processed_images = set()
        for ann in self.train_annotations['annotations']:
            image_id = ann['image_id']
            
            # Skip if we've already processed this image
            if image_id in processed_images:
                continue
                
            # Check category
            if category_ids and ann['category_id'] not in category_ids:
                continue
                
            # Check object size
            if min_obj_size and ann['area'] < min_obj_size:
                continue
                
            # Check number of instances
            if max_instances:
                image_anns = self.get_image_annotations(image_id)
                if len(image_anns) > max_instances:
                    continue
            
            # Add image info
            image_info = self.get_image_info(image_id)
            if image_info:
                suitable_images.append({
                    'image_info': image_info,
                    'annotations': self.get_image_annotations(image_id)
                })
                processed_images.add(image_id)
        
        return suitable_images

    def validate_annotation(self, annotation: Dict) -> bool:
        """Validate if annotation is suitable for inpainting tests."""
        # Check if annotation has required fields
        required_fields = ['bbox', 'segmentation', 'area']
        if not all(field in annotation for field in required_fields):
            return False
            
        # Check if bbox is valid
        bbox = annotation['bbox']
        if len(bbox) != 4 or any(x <= 0 for x in bbox[2:]):  # width and height should be positive
            return False
            
        # Check if segmentation is valid
        if not annotation['segmentation'] or not isinstance(annotation['segmentation'], list):
            return False
            
        # Check minimum area
        if annotation['area'] < 100:  # minimum area threshold
            return False
            
        return True

    def get_annotation_metadata(self, annotation: Dict) -> Dict:
        """Get metadata about the annotation useful for inpainting."""
        bbox = annotation['bbox']
        return {
            'area': annotation['area'],
            'bbox_width': bbox[2],
            'bbox_height': bbox[3],
            'aspect_ratio': bbox[2] / bbox[3] if bbox[3] > 0 else 0,
            'center_point': [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2],
            'category': self.get_category_info(annotation['category_id'])
        }