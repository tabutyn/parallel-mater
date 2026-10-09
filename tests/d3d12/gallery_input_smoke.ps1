# SPDX-License-Identifier: MIT
param(
    [string]$Executable = "$PSScriptRoot\..\..\build-d3d12-full\Release\parallel-mater-d3d12-gallery.exe",
    [string]$Output = "$PSScriptRoot\..\..\build-d3d12-full\input-smoke"
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GalleryInputSmoke {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint key, uint mapType);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rect rect);
}
'@
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$stdoutPath = Join-Path $Output 'stdout.log'
$process = Start-Process -FilePath $Executable -ArgumentList '--substeps', '1' -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError (Join-Path $Output 'stderr.log')
try {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 100
        $process.Refresh()
        if ($process.HasExited) { throw 'Viewer exited before opening its window' }
    } while (($process.MainWindowHandle -eq 0 -or $process.MainWindowTitle -notlike 'RIGID BODY*') -and (Get-Date) -lt $deadline)
    if ($process.MainWindowHandle -eq 0) { throw 'Viewer did not open a window' }
    $window = $process.MainWindowHandle
    [GalleryInputSmoke]::SetForegroundWindow($window) | Out-Null
    function Set-Key([uint32]$key,[bool]$pressed) {
        $scan = [GalleryInputSmoke]::MapVirtualKey($key, 0)
        if ($key -ge 0x25 -and $key -le 0x28) { $scan = $scan -bor 0x100 }
        if ($pressed) {
            [GalleryInputSmoke]::PostMessage($window, 0x100, [IntPtr]$key, [IntPtr](1 -bor ($scan -shl 16))) | Out-Null
        } else {
            [GalleryInputSmoke]::PostMessage($window, 0x101, [IntPtr]$key, [IntPtr]([int64]0xC0000001L -bor ($scan -shl 16))) | Out-Null
        }
    }
    function Send-Key([uint32]$key) {
        Set-Key $key $true
        Set-Key $key $false
        Start-Sleep -Milliseconds 350
    }
    function Save-Window([string]$name) {
        $rect = New-Object GalleryInputSmoke+Rect
        [GalleryInputSmoke]::GetWindowRect($window, [ref]$rect) | Out-Null
        $bitmap = New-Object Drawing.Bitmap ($rect.Right-$rect.Left), ($rect.Bottom-$rect.Top)
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
            $bitmap.Save((Join-Path $Output "$name.png"), [Drawing.Imaging.ImageFormat]::Png)
        } finally { $graphics.Dispose(); $bitmap.Dispose() }
    }
    Set-Key 0x27 $true # Hold Right Arrow in the sphere scene.
    Start-Sleep -Milliseconds 700
    Set-Key 0x27 $false
    $deadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $deadline -and
           ((Get-Content -LiteralPath $stdoutPath -Raw) -notmatch 'Arrow input: right=1 up=0')) {
        Start-Sleep -Milliseconds 100
    }
    if ((Get-Content -LiteralPath $stdoutPath -Raw) -notmatch 'Arrow input: right=1 up=0') {
        throw 'Held Right Arrow did not reach rigid-scene physics'
    }
    Send-Key 0x50 # Pause physics, keeping screenshots stable.
    Send-Key 0x46 # F
    Save-Window 'fps'
    Send-Key 0x09 # Tab
    Save-Window 'catalog'
    Send-Key 0x28 # Down to Fixed
    Send-Key 0x28 # Down to Point
    Send-Key 0x0D # Enter
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 100
        $process.Refresh()
        if ($process.HasExited) { throw 'Viewer exited while switching scenes' }
    } while ($process.MainWindowTitle -notlike 'CONSTRAINT: POINT*' -and (Get-Date) -lt $deadline)
    if ($process.MainWindowTitle -notlike 'CONSTRAINT: POINT*') { throw 'Tab/Down/Enter did not load the point scene' }
    Send-Key 0x50 # Unpause.
    Send-Key 0x20 # Space releases Point constraints.
    $deadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $deadline -and
           ((Get-Content -LiteralPath $stdoutPath -Raw) -notmatch 'Scene action: constraints released')) {
        Start-Sleep -Milliseconds 100
    }
    if ((Get-Content -LiteralPath $stdoutPath -Raw) -notmatch 'Scene action: constraints released') {
        throw 'Space did not release Point constraints'
    }
    Send-Key 0x50 # Pause again.
    Send-Key 0x09 # Reopen catalog at Point.
    Send-Key 0x28 # Down to Hinge + Slider.
    Send-Key 0x0D # Enter.
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 100
        $process.Refresh()
        if ($process.HasExited) { throw 'Viewer exited while switching scenes' }
    } while ($process.MainWindowTitle -notlike 'CONSTRAINT: HINGE*' -and (Get-Date) -lt $deadline)
    if ($process.MainWindowTitle -notlike 'CONSTRAINT: HINGE*') { throw 'Point-to-Hinge scene switch failed' }
    Start-Sleep -Milliseconds 350
    Save-Window 'switched-scene'
    Send-Key 0x1B # Escape
    if (!$process.WaitForExit(10000)) { throw 'Escape did not close the viewer' }
    if ($process.ExitCode -ne 0) { throw "Viewer exited with code $($process.ExitCode)" }
    Write-Output "Real window key/scene-switch test passed. Captures: $Output"
} finally {
    $process.Refresh()
    if (!$process.HasExited) {
        $process.CloseMainWindow() | Out-Null
        if (!$process.WaitForExit(5000)) { $process.Kill() }
    }
    $process.Dispose()
}
