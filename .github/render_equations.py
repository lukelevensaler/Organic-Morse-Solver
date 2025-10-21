#!/usr/bin/env python3
import os
import re
import hashlib
import subprocess

README = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'README.md'))
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))

# Read README content
with open(README, "r", encoding="utf-8") as f:
    text = f.read()

# Find all LaTeX math blocks delimited by $$
pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
matches = pattern.findall(text)

for eq in matches:
    eq_stripped = eq.strip()
    
    # Create a deterministic filename based on hash (avoids collisions)
    eq_hash = hashlib.sha1(eq_stripped.encode()).hexdigest()[:8]
    svg_filename = f"eq_{eq_hash}.svg"
    svg_path = os.path.join(ASSETS_DIR, svg_filename)

    print(f"Rendering: {eq_stripped} → {svg_filename}")

    # Render to SVG using latex2svg
    try:
        result = subprocess.run(
            ["latex2svg", "-", "-o", svg_path],
            input=eq_stripped.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error rendering equation: {eq_stripped}")
        print(e.stderr.decode())
        continue

    # Replace the LaTeX block in README with the image link
    img_link = f"![]({ASSETS_DIR}/{svg_filename})"
    text = text.replace(f"$${eq}$$", img_link)

# Save updated README
with open(README, "w", encoding="utf-8") as f:
    f.write(text)

print("All equations rendered and README updated.")
