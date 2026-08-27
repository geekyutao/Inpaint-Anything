# run_inpaint_tests.py
import os
import json
import argparse
from pathlib import Path
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
# run_inpaint_tests.py
from utils.coco_utils import COCOAnnotationReader

class InpaintTester:
    """Handles execution of inpainting tests."""
    
    def __init__(self, coco_dir: Path):
        self.coco_reader = COCOAnnotationReader(coco_dir)
        self.tasks = {
            "1": {
                "name": "Object Removal",
                "subtasks": {
                    "1": "Answer Consistency",
                    "2": "Focus Preservation",
                    "3": "Relationship Tests"
                },
                "script": "remove_anything.py"
            },
            "2": {
                "name": "Object Replacement",
                "subtasks": {
                    "1": "Context Adaptation",
                    "2": "Event Coherence",
                    "3": "Scale Relations"
                },
                "script": "replace_anything.py"
            },
            "3": {
                "name": "Background",
                "subtasks": {
                    "1": "Environmental",
                    "2": "Temporal",
                    "3": "Weather Effects"
                },
                "script": "fill_anything.py"
            }
        }

    def show_menu(self) -> None:
        """Display main task selection menu."""
        print("\n=== Inpaint Anything Testing ===")
        print("Available Tasks:")
        for key, task in self.tasks.items():
            print(f"{key}. {task['name']}")
        print("q. Quit")

    def show_subtasks(self, task_id: str) -> None:
        """Display subtasks for selected task."""
        print(f"\n=== {self.tasks[task_id]['name']} Subtasks ===")
        for key, subtask in self.tasks[task_id]['subtasks'].items():
            print(f"{key}. {subtask}")
        print("b. Back to main menu")

    def load_test_cases(self, task: str, subtask: str) -> List[Dict]:
        """Load test cases from configuration."""
        config_path = Path(f"test_cases/{task}/{subtask}/test_config.json")
        if not config_path.exists():
            return []
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('test_cases', [])

    def execute_test(self, task_id: str, subtask_id: str, test_case: Dict) -> Dict:
        """Execute a single test case."""
        task = self.tasks[task_id]['name'].lower().replace(" ", "_")
        subtask = self.tasks[task_id]['subtasks'][subtask_id].lower().replace(" ", "_")
        script = self.tasks[task_id]['script']

        # Get COCO data
        image_info = self.coco_reader.get_image_info(test_case['image_id'])
        annotation = self.coco_reader.get_annotation(test_case['annotation_id'])
        
        if not image_info or not annotation:
            return {
                'status': 'failed',
                'error': 'Image or annotation not found in COCO dataset'
            }

        # Get points for testing
        points = self.coco_reader.get_points_from_annotation(annotation)
        if not points:
            return {
                'status': 'failed',
                'error': 'Could not extract points from annotation'
            }

        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(f"results/{task}/{subtask}/run_{timestamp}")
        result_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "python", script,
            "--input_img", str(image_info['image_path']),
            "--coords_type", "key_in",
            "--point_coords", str(points[0][0]), str(points[0][1]),
            "--point_labels", "1",
            "--output_dir", str(result_dir),
            "--sam_model_type", test_case.get('sam_model_type', 'vit_h'),
            "--sam_ckpt", test_case.get('sam_ckpt', 'sam_vit_h_4b8939.pth')
        ]

        # Add operation-specific parameters
        if task != "object_removal":
            if 'text_prompt' not in test_case:
                return {
                    'status': 'failed',
                    'error': 'Text prompt required for this operation'
                }
            cmd.extend(["--text_prompt", test_case['text_prompt']])

        if 'dilate_kernel_size' in test_case:
            cmd.extend(["--dilate_kernel_size", str(test_case['dilate_kernel_size'])])

        # Save test metadata
        metadata = {
            'test_case': test_case,
            'coco_image_info': image_info,
            'coco_annotation': annotation,
            'points_used': points,
            'command': ' '.join(cmd),
            'timestamp': timestamp
        }
        with open(result_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Execute test
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return {
                'status': 'success',
                'result_dir': str(result_dir),
                'metadata': metadata
            }
        except subprocess.CalledProcessError as e:
            return {
                'status': 'failed',
                'error': str(e),
                'stderr': e.stderr,
                'metadata': metadata
            }

    def run(self) -> None:
        """Main interaction loop."""
        while True:
            self.show_menu()
            choice = input("\nSelect a task (or 'q' to quit): ").lower()

            if choice == 'q':
                break
            elif choice in self.tasks:
                while True:
                    self.show_subtasks(choice)
                    subtask = input("\nSelect a subtask (or 'b' for back): ").lower()

                    if subtask == 'b':
                        break
                    elif subtask in self.tasks[choice]['subtasks']:
                        # Load and show test cases
                        task_name = self.tasks[choice]['name'].lower().replace(" ", "_")
                        subtask_name = self.tasks[choice]['subtasks'][subtask].lower().replace(" ", "_")
                        test_cases = self.load_test_cases(task_name, subtask_name)

                        if not test_cases:
                            print(f"\nNo test cases found for {task_name}/{subtask_name}")
                            continue

                        print("\nAvailable test cases:")
                        for idx, test_case in enumerate(test_cases, 1):
                            image_info = self.coco_reader.get_image_info(test_case['image_id'])
                            if image_info:
                                print(f"{idx}. {test_case['name']} (Image: {image_info['file_name']})")

                        try:
                            test_idx = int(input("\nSelect test case number: ")) - 1
                            if 0 <= test_idx < len(test_cases):
                                result = self.execute_test(choice, subtask, test_cases[test_idx])
                                if result['status'] == 'success':
                                    print(f"\nTest completed successfully!")
                                    print(f"Results saved in: {result['result_dir']}")
                                else:
                                    print(f"\nTest failed: {result['error']}")
                            else:
                                print("Invalid test case selection!")
                        except ValueError:
                            print("Invalid input! Please enter a number.")