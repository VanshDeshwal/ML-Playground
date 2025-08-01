# Code Snippet Service - Extracts code snippets from core implementations
import os
import re
from typing import Optional, Dict, Any
from pathlib import Path

class SnippetService:
    """Service to extract code snippets from core algorithm implementations"""
    
    def __init__(self):
        # Path to core folder relative to backend
        self.core_path = Path(__file__).parent.parent.parent / "core"
    
    def get_algorithm_snippet(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract code snippet for a specific algorithm
        
        Args:
            algorithm_id: The algorithm identifier (e.g., 'linear_regression')
            
        Returns:
            Dictionary containing code snippet and metadata, or None if not found
        """
        try:
            # Map algorithm_id to core file
            file_mapping = {
                'linear_regression': 'linear_regression.py',
                # Add more mappings as you create more core algorithms
            }
            
            if algorithm_id not in file_mapping:
                return None
            
            file_path = self.core_path / file_mapping[algorithm_id]
            
            if not file_path.exists():
                return None
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract snippet between markers
            snippet = self._extract_snippet(content)
            
            if snippet:
                return {
                    'algorithm_id': algorithm_id,
                    'filename': file_mapping[algorithm_id],
                    'code': snippet,
                    'language': 'python',
                    'description': f'Core implementation of {algorithm_id.replace("_", " ").title()}'
                }
            
            return None
            
        except Exception as e:
            print(f"Error extracting snippet for {algorithm_id}: {e}")
            return None
    
    def _extract_snippet(self, content: str) -> Optional[str]:
        """
        Extract code between SNIPPET-START and SNIPPET-END markers
        
        Args:
            content: Full file content
            
        Returns:
            Code snippet between markers, or None if markers not found
        """
        try:
            # Pattern to match snippet markers (more flexible to handle labels)
            pattern = r'#\s*===\s*SNIPPET-START.*?===.*?\n(.*?)#\s*===\s*SNIPPET-END.*?==='
            
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                snippet = match.group(1)
                # Clean up the snippet (remove extra whitespace, but preserve indentation)
                lines = snippet.split('\n')
                
                # Remove empty lines at start and end
                while lines and not lines[0].strip():
                    lines.pop(0)
                while lines and not lines[-1].strip():
                    lines.pop()
                
                if lines:
                    # Find minimum indentation (excluding empty lines)
                    min_indent = min(len(line) - len(line.lstrip()) 
                                   for line in lines if line.strip())
                    
                    # Remove common indentation
                    cleaned_lines = []
                    for line in lines:
                        if line.strip():  # Non-empty line
                            cleaned_lines.append(line[min_indent:])
                        else:  # Empty line
                            cleaned_lines.append('')
                    
                    return '\n'.join(cleaned_lines)
            
            return None
            
        except Exception as e:
            print(f"Error parsing snippet: {e}")
            return None
    
    def get_available_snippets(self) -> Dict[str, Any]:
        """
        Get list of all available code snippets
        
        Returns:
            Dictionary with algorithm IDs and their snippet metadata
        """
        available = {}
        
        # Currently supported algorithms
        algorithm_ids = ['linear_regression']  # Expand this as you add more
        
        for algo_id in algorithm_ids:
            snippet_info = self.get_algorithm_snippet(algo_id)
            if snippet_info:
                available[algo_id] = {
                    'filename': snippet_info['filename'],
                    'description': snippet_info['description'],
                    'available': True
                }
            else:
                available[algo_id] = {
                    'filename': f'{algo_id}.py',
                    'description': f'{algo_id.replace("_", " ").title()} implementation',
                    'available': False
                }
        
        return available

# Global instance
snippet_service = SnippetService()
