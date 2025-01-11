import os
import sys

# Get the current directory path
current_dir = os.path.dirname(os.path.realpath(__file__))

# Set the path to the src directory
src_dir = os.path.join(current_dir, "src")

# Append the src directory path to the system path
sys.path.append(src_dir)

# Set the path to the tests directory
tests_dir = os.path.join(current_dir, "tests")

# Append the tests directory path to the system path
sys.path.append(tests_dir)
