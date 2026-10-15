# SPDX-License-Identifier: MIT
param(
    [string]$Executable = "$PSScriptRoot\..\..\build\Release\Parallel-Mater.exe",
    [string]$Output = "$PSScriptRoot\..\..\build\input-smoke"
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
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window, IntPtr deviceContext, uint flags);
    [DllImport("kernel32.dll")] public static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);
}
'@
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$stdoutPath = Join-Path $Output 'stdout.log'
$stderrPath = Join-Path $Output 'stderr.log'
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $Executable -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$processHandle = [IntPtr]::Zero
try {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 100
        $process.Refresh()
        if ($process.HasExited) { throw 'Viewer exited before opening its window' }
    } while ($process.MainWindowHandle -eq 0 -and (Get-Date) -lt $deadline)
    if ($process.MainWindowHandle -eq 0) { throw 'Viewer did not open a window' }
    $window = $process.MainWindowHandle
    $processHandle = $process.Handle
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
        Start-Sleep -Milliseconds 100
        Set-Key $key $false
        Start-Sleep -Milliseconds 250
    }
    function Save-Window([string]$name) {
        $rect = New-Object GalleryInputSmoke+Rect
        [GalleryInputSmoke]::GetWindowRect($window, [ref]$rect) | Out-Null
        $bitmap = New-Object Drawing.Bitmap ($rect.Right-$rect.Left), ($rect.Bottom-$rect.Top)
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        try {
            $deviceContext = $graphics.GetHdc()
            try {
                # PW_RENDERFULLCONTENT captures this HWND even if another
                # process takes the foreground while the smoke test runs.
                if (![GalleryInputSmoke]::PrintWindow($window, $deviceContext, 2)) {
                    throw "Could not capture gallery window $name"
                }
            } finally { $graphics.ReleaseHdc($deviceContext) }
            $bitmap.Save((Join-Path $Output "$name.png"), [Drawing.Imaging.ImageFormat]::Png)
        } finally { $graphics.Dispose(); $bitmap.Dispose() }
    }
    Send-Key 0x46 # F: enable live timing/contact collection.
    Start-Sleep -Milliseconds 1500
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited after enabling timing with F' }
    Save-Window 'fps'
    Send-Key 0x09 # Tab
    Save-Window 'catalog'
    Send-Key 0x28 # Down to Fixed
    Send-Key 0x28 # Down to Point
    Send-Key 0x0D # Enter
    Start-Sleep -Milliseconds 1500
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited while switching to the point scene' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -match 'Scene switch failed') {
        throw 'Tab/Down/Enter failed to load the point scene'
    }
    Send-Key 0x20 # Space releases Point constraints.
    Start-Sleep -Milliseconds 1500
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited while releasing Point constraints' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -match 'constraint failed') {
        throw 'Space failed to release Point constraints'
    }
    Send-Key 0x09 # Reopen catalog at Point.
    Send-Key 0x28 # Down to Hinge + Slider.
    Send-Key 0x0D # Enter.
    Start-Sleep -Milliseconds 1500
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited while switching to the hinge scene' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -match 'Scene switch failed') {
        throw 'Point-to-Hinge scene switch failed'
    }
    Save-Window 'switched-scene'
    Send-Key 0x1B # Escape
    if (!$process.WaitForExit(5000)) {
        # A posted key can race a long GPU frame; retry after the event loop
        # has had a full timeout window to drain.
        Send-Key 0x1B
        if (!$process.WaitForExit(10000)) { throw 'Escape did not close the viewer' }
    }
    # PowerShell 5 can leave ExitCode uncached when Start-Process redirects
    # output. Complete asynchronous stream handling, then refresh the handle.
    $process.WaitForExit()
    $process.Refresh()
    [uint32]$exitCode = 0
    if (![GalleryInputSmoke]::GetExitCodeProcess($processHandle, [ref]$exitCode)) {
        throw 'Could not read the viewer exit code'
    }
    if ($exitCode -ne 0) { throw "Viewer exited with code $exitCode" }
    Write-Output "Real window key/scene-switch test passed. Captures: $Output"
} finally {
    $process.Refresh()
    if (!$process.HasExited) {
        $process.CloseMainWindow() | Out-Null
        if (!$process.WaitForExit(5000)) { $process.Kill() }
    }
    $process.Dispose()
}
