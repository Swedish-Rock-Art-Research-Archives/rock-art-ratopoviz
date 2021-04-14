
# Ratopoviz

Rock Art Topographical Visualization.

Tool that reads 3D mesh data and generate images visualizing the content. To run, setup **config.json** with the correct paths and run **main.py**.

This is the code neccessary to create the ratopoviz executable file (see the link below for a pre-packaged executable file).

### Files

- **config.json** - Configuration file where paths and settings are defined.
- **main.py** - Main script for running the tool.
- **pc_image.py** - Methods for processing image data.
- **pc_mesh.py** - Methods for processing mesh data.
- **project_back.py** - Example of how to project back the generated image on a 3D mesh.

Link to a executable file for ratopoviz and the necessary folder structure:

[link to dropbox](https://www.dropbox.com/s/nflb82fdfgdrqcx/ratopoviz.zip?dl=1)

Drop laser scan files (format: .ply, .stl) into the folder "original", then execute ratopoviz.exe. Output in the folder "processed". There is a known error that will end the process without producing the visualizations, and we are uploading an update soon.
