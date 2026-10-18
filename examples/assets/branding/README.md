# Gallery branding

`logo.png` is the approved ParallelMater artwork. The Linux desktop entry uses
that source image and pairs it with the `parallel-mater` window class.
`ParallelMater.ico` supplies Windows Explorer, Start menu, window, and taskbar
icons. The Windows resource is embedded in the single gallery executable,
including builds with the optional CUDA mode. No runtime checkout access is
needed to load a Windows icon.

Generated platform containers are committed so building the app needs no
image tools. To regenerate on macOS after replacing `logo.png`, install
ImageMagick and run:

```sh
bash scripts/Generate-GalleryIcons.sh
```

The conversion preserves the full artwork, padding nonsquare sources rather
than cropping them. Windows contains 16, 24, 32, 48, 64, 128 and 256 pixel
images.

Deploy Windows with
`cmake --build build-gallery --config Release --target install-windows-gallery`.
Reopen a running gallery after deployment to load the updated resources.
