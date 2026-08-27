# test_weather.py
from MR_testing.test_cases.background.base_background import BaseBackgroundTest
from pathlib import Path
import json
import argparse
from datetime import datetime
from MR_testing.models_to_test.init_models import load_llava_model
import torch
import random
import os

class WeatherTestRunner:
    def __init__(self, config_path: str, llm_evaluator=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize with provided evaluator (will be LLaVA)
        self.llm_evaluator = llm_evaluator
        
        # Initialize base test
        self.base_test = BaseBackgroundTest(
            config=self.config['default_params'],
            llm_evaluator=self.llm_evaluator
        )

    def get_random_image_and_annotations(self, images_dir, annotations_path):
        """Select a random image from the directory and get its annotations."""
        # List all images in the directory
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if not images:
            raise ValueError("No images found in the specified directory.")

        # Select a random image
        selected_image = random.choice(images)
        image_path = os.path.join(images_dir, selected_image)

        # Get the image ID from the file name (assuming the file name is the image ID)
        image_id = int(os.path.splitext(selected_image)[0])

        # Get coordinates from annotations
        point_coords = self.base_test.get_coordinates_from_annotations(annotations_path, image_id)

        return image_path, point_coords, image_id

    def run_test(self, images_dir: str, annotations_path: str, selected_variations: list = None):
        """Run weather variation tests."""
        if not self.llm_evaluator:
            print("Warning: No LLM evaluator provided. Please initialize with LLaVA before running tests.")
            return

        # Get a random image and its annotations
        image_path, point_coords, image_id = self.get_random_image_and_annotations(images_dir, annotations_path)

        # Get variations to test
        variations = self.config['variations']
        if selected_variations:
            variations = {k: v for k, v in variations.items() if k in selected_variations}

        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nStarting Weather Variation Tests")
        print(f"Image: {image_path}")
        print(f"Testing {len(variations)} weather variations")
        print("-" * 50)

        # Test each variation
        for weather_type, weather_config in variations.items():
            print(f"\nTesting {weather_type} conditions:")
            print(f"Description: {weather_config['description']}")
            
            test_case = {
                'test_type': 'weather',
                'image_path': image_path,
                'point_coords': point_coords,
                'prompt': weather_config['prompt']
            }
            
            result = self.base_test.execute_test(test_case)
            
            if result['status'] == 'success':
                results.append({
                    'weather_type': weather_type,
                    'result': result
                })
                print(f"Original object: {result['original_object']}")
                print(f"After weather change: {result['result_object']}")
                print(f"Identity preserved: {'✓' if result['identity_preserved'] else '✗'}")
            else:
                print(f"Failed: {result['error']}")

        # Save results
        results_dir = Path("results/weather_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"weather_test_results_{timestamp}.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'image_path': image_path,
                'variations_tested': list(variations.keys()),
                'results': results
            }, f, indent=2)

        # Print summary
        print("\nTest Summary:")
        print("-" * 50)
        successful_tests = sum(1 for r in results if r['result']['identity_preserved'])
        print(f"Total variations tested: {len(results)}")
        print(f"Successful identity preservation: {successful_tests}/{len(results)}")
        print(f"Results saved to: {results_dir}/weather_test_results_{timestamp}.json")

def main():
    parser = argparse.ArgumentParser(description="Run weather variation tests")
    parser.add_argument("--images_dir", required=True, help="Path to directory containing images")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations file")
    parser.add_argument("--config", default="/workspaces/Advanced-MRs-for-VLMs/MR_testing/test_cases/background/configs/weather_config.json",
                       help="Path to weather test configuration")
    parser.add_argument("--variations", nargs="+", 
                       choices=["rainy", "snowy", "foggy", "sunny", "stormy"],
                       help="Specific weather variations to test")

    args = parser.parse_args()

    # Load LLaVa model
    llava_evaluator = load_llava_model("/workspaces/Advanced-MRs-for-VLMs/MR_testing/models_to_test/llava_config.json")
    
    # Create and run tests
    tester = WeatherTestRunner(args.config, llava_evaluator)
    tester.run_test(args.images_dir, args.annotations, args.variations)

if __name__ == "__main__":
    main()