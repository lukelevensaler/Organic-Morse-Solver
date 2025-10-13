"""
Morse Solver - Succinct entry point for all algorithm operations
Usage: 
  ./morse stretches              # Show available bond stretches
  ./morse compute [args...]      # Run full algorithm with interactive or CLI args
  ./morse --help                 # Show all options
"""
import sys
import os

# Ensure we can import from the current directory
if __name__ == "__main__":
    # Add current directory to path so we can import morse_solver modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import and run the CLI app
    try:
        from cli import app
        app()
    except ImportError as e:
        print(f"Error importing morse solver modules: {e}")
        print("Make sure you're running from the morse_solver directory with conda environment activated")
        sys.exit(1)