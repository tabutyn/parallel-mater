# SPDX-License-Identifier: MIT
[CmdletBinding(SupportsShouldProcess=$true)]
param([switch]$Quiet)
$ErrorActionPreference = 'Stop'
$expectedRoot = [IO.Path]::GetFullPath((Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Parallel-Mater'))
$installRoot = [IO.Path]::GetFullPath($PSScriptRoot)
if ($installRoot -ne $expectedRoot) { throw 'Run the uninstaller from the installed Parallel-Mater directory.' }
foreach ($directory in @($installRoot,(Join-Path $installRoot 'assets'))) {
    if ((Test-Path -LiteralPath $directory) -and ((Get-Item -LiteralPath $directory).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing a linked installation directory: $directory"
    }
}
$manifestPath = Join-Path $installRoot 'installation.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.application -ne 'Parallel-Mater' -or $manifest.directory -ne $installRoot) { throw 'Invalid installation ownership marker.' }
$targets = @($manifest.files | ForEach-Object {
    $target = [IO.Path]::GetFullPath((Join-Path $installRoot $_))
    if (!$target.StartsWith($installRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe installation manifest path.' }
    $target
})
$running = Get-Process -Name 'Parallel-Mater','parallel-mater-d3d12-gallery' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and [IO.Path]::GetDirectoryName($_.Path) -eq $installRoot }
if ($running) { throw 'Close Parallel-Mater before uninstalling it.' }
if (!$PSCmdlet.ShouldProcess($installRoot,'Uninstall Parallel-Mater; preserve user captures and logs')) { return }
if (!$Quiet) {
    Add-Type -AssemblyName System.Windows.Forms
    $answer = [Windows.Forms.MessageBox]::Show('Uninstall Parallel-Mater? Your captures and logs will be kept.',
        'Parallel-Mater','YesNo','Question')
    if ($answer -ne 'Yes') { return }
}
# Delete only recorded package files. Never recursively remove an installation
# directory: captures, user-added files, and other unrecorded data must survive.
foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force }
}
Remove-Item -LiteralPath $manifestPath -Force
foreach ($directory in @((Join-Path $installRoot 'assets'),$installRoot)) {
    if ((Test-Path -LiteralPath $directory) -and !(Get-ChildItem -LiteralPath $directory -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $directory
    }
}
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Parallel-Mater.lnk'
if (Test-Path -LiteralPath $shortcutPath) {
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
    if ($shortcut.TargetPath -eq (Join-Path $installRoot 'Parallel-Mater.exe')) { Remove-Item -LiteralPath $shortcutPath }
}
Remove-Item -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Parallel-Mater'
Write-Output 'Parallel-Mater uninstalled. User captures and logs were preserved.'
