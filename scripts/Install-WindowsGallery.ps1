# SPDX-License-Identifier: MIT
[CmdletBinding()]
param(
    [string]$BuildDirectory,
    [string]$RuntimeDirectory,
    [string]$Version = '0.1.0'
)
$ErrorActionPreference = 'Stop'
if (![Environment]::Is64BitOperatingSystem) { throw 'Parallel-Mater requires 64-bit Windows.' }
$sourceRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
if (!$BuildDirectory) { $BuildDirectory = Join-Path $sourceRoot 'build-d3d12-full\Release' }
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$installRoot = [IO.Path]::GetFullPath((Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Parallel-Mater'))
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Parallel-Mater.lnk'
$registryPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Parallel-Mater'
$manifestPath = Join-Path $installRoot 'installation.json'
foreach ($directory in @($installRoot,(Join-Path $installRoot 'assets'))) {
    if ((Test-Path -LiteralPath $directory) -and ((Get-Item -LiteralPath $directory).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing a linked installation directory: $directory"
    }
}
if (Test-Path -LiteralPath $registryPath) {
    if ((Get-ItemProperty -LiteralPath $registryPath).InstallLocation -ne $installRoot) {
        throw 'An unrelated Parallel-Mater installation is already registered.'
    }
}
$previousFiles = @()
if (Test-Path -LiteralPath $installRoot) {
    if (!(Test-Path -LiteralPath $manifestPath)) {
        throw "Destination exists but is not a managed Parallel-Mater installation: $installRoot"
    }
    $previous = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($previous.application -ne 'Parallel-Mater' -or $previous.directory -ne $installRoot) {
        throw 'Installation ownership marker is invalid.'
    }
    $previousFiles = @($previous.files)
}
$running = Get-Process -Name 'Parallel-Mater','parallel-mater-d3d12-gallery' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and [IO.Path]::GetDirectoryName($_.Path) -eq $installRoot }
if ($running) { throw 'Close the installed Parallel-Mater gallery before updating it.' }
if (Test-Path -LiteralPath $shortcutPath) {
    $existing = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
    if ($existing.TargetPath -ne (Join-Path $installRoot 'Parallel-Mater.exe')) {
        throw "An unrelated Start menu shortcut already exists: $shortcutPath"
    }
}
$files = [ordered]@{}
foreach ($name in @('Parallel-Mater.exe','parallel-mater-d3d12-gallery.exe')) {
    $files[$name] = (Resolve-Path -LiteralPath (Join-Path $BuildDirectory $name)).Path
}
foreach ($dll in Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.dll' -File) {
    $files[$dll.Name] = $dll.FullName
}
if ($RuntimeDirectory) {
    foreach ($dll in Get-ChildItem -LiteralPath $RuntimeDirectory -Filter '*.dll' -File) {
        $files[$dll.Name] = $dll.FullName
    }
}
foreach ($name in @('RigidBody','ConstraintFixed','ConstraintPoint','ConstraintHinge',
                    'ConstraintPiston','ConstraintGeneric','ConstraintMotorSpring','DumpTruck')) {
    $files["assets\$name.glb"] = (Resolve-Path -LiteralPath "$sourceRoot\examples\assets\$name.glb").Path
}
$files['LICENSE'] = "$sourceRoot\LICENSE"
$files['D3D12.md'] = "$sourceRoot\docs\D3D12.md"
$files['Uninstall.ps1'] = "$PSScriptRoot\Uninstall-WindowsGallery.ps1"
# Resolve every input before making installation changes.
foreach ($source in $files.Values) { if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing package file: $source" } }
New-Item -ItemType Directory -Path $installRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $installRoot 'assets') -Force | Out-Null
$manifest = [ordered]@{ application='Parallel-Mater'; directory=$installRoot; version=$Version;
    files=@(@($previousFiles) + @($files.Keys) | Sort-Object -Unique) }
# Keep the ownership marker even if a copy fails, so rerunning can repair it.
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
foreach ($name in $files.Keys) { Copy-Item -LiteralPath $files[$name] -Destination (Join-Path $installRoot $name) -Force }
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $installRoot 'Parallel-Mater.exe'
$shortcut.WorkingDirectory = $installRoot
$shortcut.Description = 'Parallel-Mater - DirectCompute Physics Gallery'
$shortcut.IconLocation = $shortcut.TargetPath + ',0'
$shortcut.Save()
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$uninstall = '"' + $powershell + '" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $installRoot 'Uninstall.ps1') + '"'
New-Item -Path $registryPath -Force | Out-Null
foreach ($entry in @{
    DisplayName='Parallel-Mater'; DisplayVersion=$Version; Publisher='Parallel-Mater';
    InstallLocation=$installRoot; DisplayIcon=(Join-Path $installRoot 'Parallel-Mater.exe');
    UninstallString=$uninstall; QuietUninstallString=($uninstall + ' -Quiet');
    URLInfoAbout='https://github.com/tabutyn/parallel-mater'; InstallDate=(Get-Date -Format yyyyMMdd)
}.GetEnumerator()) {
    New-ItemProperty -Path $registryPath -Name $entry.Key -Value $entry.Value -PropertyType String -Force | Out-Null
}
$size = [int][Math]::Ceiling((($files.Values | ForEach-Object { (Get-Item -LiteralPath $_).Length } | Measure-Object -Sum).Sum)/1KB)
foreach ($entry in @{ NoModify=1; NoRepair=1; EstimatedSize=$size }.GetEnumerator()) {
    New-ItemProperty -Path $registryPath -Name $entry.Key -Value $entry.Value -PropertyType DWord -Force | Out-Null
}
Write-Output "Installed Parallel-Mater: $installRoot"
Write-Output "Start menu: $shortcutPath"
