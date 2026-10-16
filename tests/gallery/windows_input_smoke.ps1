# SPDX-License-Identifier: MIT
param(
    [string]$Executable = "$PSScriptRoot\..\..\build\Release\Parallel-Mater.exe",
    [string]$Output = "$PSScriptRoot\..\..\build\input-smoke",
    [ValidateSet('d3d12','cuda')][string]$Backend = 'd3d12',
    [ValidateRange(5,60)][int]$DwellSeconds = 5
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
    [DllImport("user32.dll")] public static extern IntPtr SendMessageTimeout(IntPtr window, uint message, IntPtr wParam, IntPtr lParam, uint flags, uint timeout, out IntPtr result);
}
'@
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$stdoutPath = Join-Path $Output 'stdout.log'
$stderrPath = Join-Path $Output 'stderr.log'
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $Executable -ArgumentList @('--backend',$Backend,'--fluid-particle-view','--dump-spheres','10','--fluid-particles','100','--brick-count','8') -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
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
    $script:sceneGeneration = 0
    function Wait-Scene([string]$name, [int]$expected = -1) {
        $readyDeadline = (Get-Date).AddMinutes(3)
        do {
            $process.Refresh()
            if ($process.HasExited) { throw "Viewer exited during $name" }
            [IntPtr]$reply = [IntPtr]::Zero
            $ready = [GalleryInputSmoke]::SendMessageTimeout($window, 0, [IntPtr]::Zero, [IntPtr]::Zero, 2, 200, [ref]$reply)
            $sceneReady = $expected -lt 0
            if (!$sceneReady) {
                $readyText = Get-Content -LiteralPath $stdoutPath -Raw
                $readyLines = @()
                if ($readyText) {
                    $readyLines = [regex]::Matches($readyText, 'Gallery scene ready index=(\d+) revision=(\d+)')
                }
                if ($readyLines.Count -gt 0) {
                    $lastReady = $readyLines[$readyLines.Count - 1]
                    $sceneReady = [int]$lastReady.Groups[1].Value -eq $expected -and [int]$lastReady.Groups[2].Value -gt $script:sceneGeneration
                }
            }
            if ($ready -ne [IntPtr]::Zero -and $sceneReady) {
                if ($expected -ge 0) { $script:sceneGeneration = [int]$lastReady.Groups[2].Value }
                break
            }
            Start-Sleep -Milliseconds 200
        } while ((Get-Date) -lt $readyDeadline)
        if ($ready -eq [IntPtr]::Zero -or !$sceneReady) { throw "Viewer did not finish loading/processing input during $name" }
        Start-Sleep -Seconds $DwellSeconds
        $process.Refresh()
        if ($process.HasExited) { throw "Viewer exited during $name dwell" }
        $errors = Get-Content -LiteralPath $stderrPath -Raw
        if ($errors -match '(?i)(failed|device lost|invalid state)') { throw "$name reported: $errors" }
        Write-Output "$Backend ${name}: $DwellSeconds seconds observed"
    }
    Wait-Scene 'startup' 0
    Send-Key 0x46 # F: show shared timing; solver workload is unchanged.
    Wait-Scene 'F timing'
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited after enabling timing with F' }
    Save-Window 'fps'
    Send-Key 0x46 # F off: native rendering must work without diagnostic readback.
    Wait-Scene 'GPU-only rendering'
    Send-Key 0x09 # Tab
    Save-Window 'catalog'
    Send-Key 0x28 # Down to Fixed
    Send-Key 0x28 # Down to Point
    Send-Key 0x0D # Enter
    Wait-Scene 'point scene' 2
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited while switching to the point scene' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -match 'Scene switch failed') {
        throw 'Tab/Down/Enter failed to load the point scene'
    }
    Send-Key 0x20 # Space releases Point constraints.
    Wait-Scene 'point release'
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited while releasing Point constraints' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -match 'constraint failed') {
        throw 'Space failed to release Point constraints'
    }
    Send-Key 0x09 # Reopen catalog at Point.
    Send-Key 0x28 # Down to Hinge + Slider.
    Send-Key 0x0D # Enter.
    Wait-Scene 'hinge scene' 3
    $process.Refresh()
    if ($process.HasExited) { throw 'Viewer exited while switching to the hinge scene' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -match 'Scene switch failed') {
        throw 'Point-to-Hinge scene switch failed'
    }
    Save-Window 'switched-scene'
    # Canonical catalog indices, identical on every backend. D3D12 currently
    # advertises rigid, liquid and cloth, but not paint/rope/soft/smoke.
    $sceneIndices = if ($Backend -eq 'cuda') { 0..27 } else { @(0,1,2,3,4,5,6,7,8,9,11,12,14) }
    $currentScene = 3
    foreach ($sceneIndex in $sceneIndices) {
        Send-Key 0x09
        $delta = $sceneIndex - $currentScene
        $key = if ($delta -ge 0) { 0x28 } else { 0x26 }
        for ($move = 0; $move -lt [Math]::Abs($delta); ++$move) { Send-Key $key }
        Send-Key 0x0D
        Wait-Scene "scene-$sceneIndex" $sceneIndex
        Save-Window "scene-$sceneIndex"
        if ($sceneIndex -in @(8,9,14)) {
            Send-Key 0x56 # V: surface, or ClothWater wireframe from particle view.
            Wait-Scene "scene-$sceneIndex-surface"
            Save-Window "scene-$sceneIndex-surface"
            Send-Key 0x56
            Wait-Scene "scene-$sceneIndex-particles"
            Send-Key 0x46
            Wait-Scene "scene-$sceneIndex-diagnostics"
            Send-Key 0x46
        }
        if ($sceneIndex -in @(11,12)) {
            Send-Key 0x56
            Wait-Scene "scene-$sceneIndex-structure"
            Save-Window "scene-$sceneIndex-structure"
            Send-Key 0x56
        }
        $currentScene = $sceneIndex
        Send-Key 0x52 # R: reset current scene without changing backend.
        Wait-Scene "scene-$sceneIndex-reset" $sceneIndex
    }
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
