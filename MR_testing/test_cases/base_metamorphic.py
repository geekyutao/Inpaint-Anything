from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image
import subprocess
from pathlib import Path

class BaseMetamorphicTest(ABC):
    def __init__(self, config: Dict[str, Any], llm_evaluator):
        self.config = config
        self.llm_evaluator = llm_evaluator

    @abstractmethod
    def execute_test(self, test_case: Dict) -> Dict[str, Any]:
        pass

    @abstractmethod
    def verify_relations(self, original: Image.Image, result: Image.Image, test_case: Dict) -> Dict[str, bool]:
        pass