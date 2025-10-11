Rainbow Perimeter frame export

Generated 2025-10-11T08:36:24.695Z
Frames: 240
Resolution: 1000x476
Frame rate: 60 fps

Use ffmpeg to assemble:
ffmpeg -framerate 60 -i frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p rainbow-output.mp4
