"""
Output management system for CNN learning experiments.

This module provides centralized output handling for plots and terminal output.
Each day gets its own out/ directory automatically.
"""

import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import inspect

class OutputManager:
    """Manages all output for CNN experiments - plots and text."""
    
    def __init__(self, base_dir: Optional[str] = None) -> None:
        """Initialize output manager with automatic day detection."""
        if base_dir is None:
            # Automatically detect which day directory we're in
            base_dir = self._detect_day_directory()
        
        self.base_dir = base_dir
        self.plots_dir = os.path.join(base_dir, "plots")
        self.output_file = os.path.join(base_dir, "out.md")
        
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize output file
        self._initialize_output_file()
    
    def _detect_day_directory(self) -> str:
        """Automatically detect which day directory we're being called from."""
        # Get the calling frame to find where we're being imported from
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling module
            while frame:
                frame = frame.f_back
                if frame and frame.f_code.co_filename != __file__:
                    # Found the calling file
                    calling_file = frame.f_code.co_filename
                    calling_dir = os.path.dirname(calling_file)
                    
                    # Check if we're in a day directory
                    dir_name = os.path.basename(calling_dir)
                    if dir_name.startswith('day') and any(c.isdigit() for c in dir_name):
                        # We're in a day directory, use it
                        return os.path.join(calling_dir, "out")
                    
                    # Check if we're in the root CNN_from_scratch directory
                    # and there are day directories around
                    if any(d.startswith('day') for d in os.listdir(calling_dir) 
                           if os.path.isdir(os.path.join(calling_dir, d))):
                        # We're in the root, create a general out directory
                        return os.path.join(calling_dir, "out")
                    
            # Fallback: create out directory in current working directory
            return "./out"
        finally:
            del frame
        
    def _initialize_output_file(self) -> None:
        """Initialize the markdown output file with header."""
        day_name = self._get_day_name()
        if not os.path.exists(self.output_file):
            # File doesn't exist, create it with header
            with open(self.output_file, 'w') as f:
                f.write(f"# CNN From Scratch - {day_name} Experiment Output\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Output directory: `{self.base_dir}`\n\n")
                f.write("---\n\n")
        else:
            # File exists, just add a session separator
            with open(self.output_file, 'a') as f:
                f.write(f"\n\n---\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
    
    def _get_day_name(self) -> str:
        """Get a readable day name from the output directory."""
        # Extract day name from path
        if 'day' in self.base_dir.lower():
            # Find the day directory name
            parts = self.base_dir.split(os.sep)
            for part in parts:
                if part.lower().startswith('day') and any(c.isdigit() for c in part):
                    return part.title()  # e.g., "day01" -> "Day01"
        return "General Experiments"
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message to both console and output file."""
        # Print to console
        print(message)
        
        # Write to file
        with open(self.output_file, 'a') as f:
            if level == "HEADER":
                f.write(f"\n## {message}\n\n")
            elif level == "SUBHEADER":
                f.write(f"\n### {message}\n\n")
            elif level == "CODE":
                f.write(f"```\n{message}\n```\n\n")
            else:
                f.write(f"{message}\n\n")
    
    def save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
        """Save the current matplotlib figure and return the path."""
        # Auto-add day prefix if not already present
        day_name = self._get_day_name().lower()
        if not filename.lower().startswith(day_name) and day_name != "general experiments":
            base_name, ext = os.path.splitext(filename)
            filename = f"{day_name}_{base_name}{ext}"
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()  # Close to free memory
        
        # Log the plot save
        relative_path = os.path.relpath(filepath, self.base_dir)
        self.log(f"![{filename}]({relative_path})", level="INFO")
        
        return filepath
    
    @contextmanager
    def experiment_section(self, title: str):
        """Context manager for experiment sections."""
        self.log(title, level="HEADER")
        try:
            yield self
        finally:
            self.log("---", level="INFO")

# Global output manager instance - automatically detects day directory
output_manager = None

def get_output_manager() -> OutputManager:
    """Get or create the output manager for the current context."""
    global output_manager
    if output_manager is None:
        output_manager = OutputManager()
    return output_manager

def create_day_output_manager(day_num: int) -> OutputManager:
    """Create an output manager for a specific day."""
    day_dir = f"day{day_num:02d}"
    current_dir = os.getcwd()
    
    # Look for the day directory
    if os.path.exists(day_dir):
        base_dir = os.path.join(day_dir, "out")
    elif os.path.exists(os.path.join("..", day_dir)):
        base_dir = os.path.join("..", day_dir, "out")
    else:
        # Create in current directory with day prefix
        base_dir = f"./{day_dir}_out"
    
    return OutputManager(base_dir)

def log_print(*args, sep: str = ' ', end: str = '\n', level: str = "INFO") -> None:
    """Enhanced print function that logs to file."""
    message = sep.join(str(arg) for arg in args) + end.rstrip()
    get_output_manager().log(message, level)

def save_plot(filename: str, **kwargs) -> str:
    """Save current plot with enhanced error handling."""
    try:
        return get_output_manager().save_plot(filename, **kwargs)
    except Exception as e:
        log_print(f"Error saving plot {filename}: {e}", level="ERROR")
        return ""

def log_array_stats(name: str, array: np.ndarray) -> None:
    """Log statistics about a numpy array."""
    stats = f"{name} stats: shape={array.shape}, min={array.min():.3f}, max={array.max():.3f}, mean={array.mean():.3f}, std={array.std():.3f}"
    log_print(stats, level="CODE")

def log_experiment_start(day: int, topic: str) -> None:
    """Log the start of a day's experiments."""
    header = f"Day {day}: {topic}"
    log_print(header, level="HEADER")
    log_print(f"Experiment started at: {datetime.now().strftime('%H:%M:%S')}")

def log_experiment_end(day: int) -> None:
    """Log the end of a day's experiments."""
    log_print(f"Day {day} experiments completed at: {datetime.now().strftime('%H:%M:%S')}")
    log_print("=" * 60)

def experiment_section(title: str):
    """Context manager for experiment sections."""
    return get_output_manager().experiment_section(title)