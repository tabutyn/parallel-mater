# Gallery branding

`logo.png` is the approved ParallelMater artwork. Both platforms use this
source: `ParallelMater.icns` supplies the macOS Finder/Dock icon, and
`ParallelMater.ico` supplies Windows Explorer, Start menu, window and taskbar
icons. The Windows resource is embedded in the D3D12 launcher and gallery,
and in the CUDA gallery when built on Windows. No runtime checkout access is
needed to load an icon.

Generated platform containers are committed so building the app needs no
image tools. To regenerate on macOS after replacing `logo.png`, install
ImageMagick and run:

```sh
bash scripts/Generate-GalleryIcons.sh
```

The conversion preserves the full artwork, padding nonsquare sources rather
than cropping them. Windows contains 16, 24, 32, 48, 64, 128 and 256 pixel
images; macOS includes standard and Retina sizes through 1024 pixels.

Deploy macOS with `cmake --build build-metal-gallery --target deploy-metal-gallery`.
Deploy Windows with `cmake --build build-d3d12-full --config Release --target install-d3d12-gallery`.
Reopen a running gallery after deployment to load the updated resources.
