# Image Quilting for Texture Synthesis and Transfer
 
run pyGameMain to start.
WASD for character movement
Arrow keys for camera movement

When game slows down, it is generating new textures in the background.




**Config file:**

ENABLE_GENERATOR_CACHING: This will enable/disable texture caching. When disabled, textures are not saved/read from file. This will drasticly slowdown the performance.

TEXTURE_GENERATOR_MAX_LIMIT: This is number of patches to be generated.

Terrain settings are unique to each texture sample.

Best values for Cobblestone:
TERRAIN_SAMPLE_PATCH_SIZE = 20
TERRAIN_QUILTING_SIZE = 2

This demo can be modded, textures can be swapped from GAME TEXTURES section. To sample other textures, delete the contents of engine/patches folder and change SAMPLE_IMAGE_PATH
