from MR_testing.test_cases.base_metamorphic import BaseMetamorphicTest
from PIL import Image
from typing import Dict, Any, Tuple
from pathlib import Path
import subprocess
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json

class BaseBackgroundTest(BaseMetamorphicTest):
    """Base class for background variation tests focusing on object identity."""
    
    def __init__(self, config: Dict[str, Any], llm_evaluator):
        super().__init__(config, llm_evaluator)
        self.replace_script = "/workspaces/Advanced-MRs-for-VLMs/replace_anything.py"
        self.sam_weights = "/workspaces/Advanced-MRs-for-VLMs/weights/mobile_sam.pt"

        # Initialize LLaVa model and processor
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    def run_replacement(self, 
                       image_path: Path, 
                       point_coords: Tuple[int, int],
                       prompt: str,
                       output_dir: Path) -> Dict[str, Any]:
        """Execute replace_anything.py with given parameters."""
        cmd = [
            "python", self.replace_script,
            "--input_img", str(image_path),
            "--coords_type", "key_in",
            "--point_coords", str(point_coords[0]), str(point_coords[1]),
            "--point_labels", "1",
            "--text_prompt", prompt,
            "--output_dir", str(output_dir),
            "--sam_model_type", "vit_h",
            "--sam_ckpt", self.sam_weights
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            result_path = output_dir / "replaced_with_mask_0.png"
            
            if result_path.exists():
                return {
                    'status': 'success',
                    'output_path': result_path,
                    'output_image': Image.open(result_path)
                }
            return {
                'status': 'failed', 
                'error': 'Output image not found'
            }
        except subprocess.CalledProcessError as e:
            return {
                'status': 'failed', 
                'error': str(e)
            }

    def get_object_description(self, image: Image.Image) -> str:
        """Get one-word object identification using LLM."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is the main object in this image? Give one word answer."},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=10)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

    def compare_identity(self, original: Image.Image, result: Image.Image) -> Dict[str, Any]:
        """Compare object identity between original and result image."""
        original_object = self.get_object_description(original)
        result_object = self.get_object_description(result)
        
        return {
            'original_object': original_object,
            'result_object': result_object,
            'identity_preserved': original_object.lower() == result_object.lower()
        }

    def get_coordinates_from_annotations(self, annotations_path, image_id):
        """Extract coordinates from COCO annotations."""
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        for annotation in annotations['annotations']:
            if annotation['image_id'] == image_id:
                bbox = annotation['bbox']
                # COCO format is [x, y, width, height]
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                return (int(x_center), int(y_center))
        
        raise ValueError(f"No annotations found for image_id {image_id}")

    def execute_test(self, test_case: Dict) -> Dict[str, Any]:
        """Execute the test and return results."""
        image_path = Path(test_case['image_path'])
        results_dir = Path(f"results/background/{test_case['test_type']}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get original image description
        original_img = Image.open(image_path)
        original_desc = self.get_object_description(original_img)

        # Apply background change
        result = self.run_replacement(
            image_path,
            test_case['point_coords'],
            test_case['prompt'],
            results_dir
        )

        if result['status'] == 'success':
            identity_check = self.compare_identity(original_img, result['output_image'])
            
            return {
                'status': 'success',
                'original_path': str(image_path),
                'result_path': str(result['output_path']),
                'original_object': identity_check['original_object'],
                'result_object': identity_check['result_object'],
                'identity_preserved': identity_check['identity_preserved']
            }
        else:
            return {
                'status': 'failed',
                'error': result['error']
            }

    def verify_relations(self, original: Image.Image, result: Image.Image, test_case: Dict) -> Dict[str, bool]:
        """Verify relations between original and result image."""
        identity_check = self.compare_identity(original, result)
        return {
            'identity_preserved': identity_check['identity_preserved']
        }