"""
Progress tracking system for Morse solver computations.

This module provides a unified progress bar system that can be used throughout
the codebase to show computation progress instead of simple time logging.
"""

import time
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich import print as rich_print

console = Console()

class MorseProgressTracker:
    """Centralized progress tracking for all Morse solver operations."""
    
    def __init__(self):
        self.progress: Optional[Progress] = None
        self.active_tasks: Dict[str, TaskID] = {}
        self.task_stack: List[str] = []
    
    def start_session(self, description: str = "Morse Solver Operations"):
        """Start a new progress tracking session."""
        if self.progress is not None:
            self.stop_session()
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        )
        self.progress.start()
        console.print(f"\n[bold green]🚀 Starting {description}[/bold green]")
        return self
    
    def stop_session(self):
        """Stop the current progress tracking session."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
        self.active_tasks.clear()
        self.task_stack.clear()
    
    def add_task(self, task_id: str, description: str, total: Optional[int] = None) -> TaskID:
        """Add a new progress task."""
        if self.progress is None:
            raise RuntimeError("Progress session not started. Call start_session() first.")
        
        task = self.progress.add_task(description, total=total)
        self.active_tasks[task_id] = task
        return task
    
    def update_task(self, task_id: str, advance: int = 1, description: Optional[str] = None):
        """Update progress on a task."""
        if self.progress is None or task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        if description:
            self.progress.update(task, advance=advance, description=description)
        else:
            self.progress.update(task, advance=advance)
    
    def complete_task(self, task_id: str, description: Optional[str] = None):
        """Mark a task as completed."""
        if self.progress is None or task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        if description:
            self.progress.update(task, description=f"✅ {description}")
        else:
            current_desc = self.progress.tasks[task].description
            self.progress.update(task, description=f"✅ {current_desc}")
        
        # Complete the task
        self.progress.update(task, completed=self.progress.tasks[task].total or 100)
    
    def set_task_total(self, task_id: str, total: int):
        """Set the total for a task (useful for indeterminate tasks that become determinate)."""
        if self.progress is None or task_id not in self.active_tasks:
            return
        
        self.progress.update(self.active_tasks[task_id], total=total)
    
    @contextmanager
    def task_context(self, task_id: str, description: str, total: Optional[int] = None):
        """Context manager for automatic task management."""
        task = self.add_task(task_id, description, total)
        self.task_stack.append(task_id)
        try:
            yield self
        except Exception as e:
            if self.progress is not None:
                self.progress.update(task, description=f"❌ {description} - Failed: {str(e)}")
            raise
        else:
            self.complete_task(task_id)
        finally:
            if task_id in self.task_stack:
                self.task_stack.remove(task_id)
    
    def print_step(self, message: str, style: str = "bold cyan"):
        """Print a step message with nice formatting."""
        console.print(f"[{style}]→ {message}[/{style}]")
    
    def print_success(self, message: str):
        """Print a success message."""
        console.print(f"[bold green]✅ {message}[/bold green]")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
    
    def print_error(self, message: str):
        """Print an error message."""
        console.print(f"[bold red]❌ {message}[/bold red]")
    
    def print_info(self, message: str):
        """Print an info message."""
        console.print(f"[blue]ℹ️  {message}[/blue]")

# Global progress tracker instance
progress_tracker = MorseProgressTracker()

# Convenience functions for common operations
def start_progress_session(description: str = "Morse Solver Operations"):
    """Start a new progress tracking session."""
    return progress_tracker.start_session(description)

def stop_progress_session():
    """Stop the current progress tracking session."""
    progress_tracker.stop_session()

@contextmanager
def progress_session(description: str = "Morse Solver Operations"):
    """Context manager for progress sessions."""
    tracker = start_progress_session(description)
    try:
        yield tracker
    finally:
        stop_progress_session()

def track_task(task_id: str, description: str, total: Optional[int] = None):
    """Context manager for tracking a single task."""
    return progress_tracker.task_context(task_id, description, total)

def print_step(message: str, style: str = "bold cyan"):
    """Print a step message."""
    progress_tracker.print_step(message, style)

def print_success(message: str):
    """Print a success message."""
    progress_tracker.print_success(message)

def print_warning(message: str):
    """Print a warning message."""
    progress_tracker.print_warning(message)

def print_error(message: str):
    """Print an error message."""
    progress_tracker.print_error(message)

def print_info(message: str):
    """Print an info message."""
    progress_tracker.print_info(message)

# Helper decorators for automatic progress tracking
def track_function_progress(task_id: str, description: str):
    """Decorator to automatically track function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with track_task(task_id, description):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_ccsd_calculation(description: str):
    """Specialized decorator for CCSD calculations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with track_task("ccsd_calc", f"CCSD(T) Calculation - {description}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator