# SPDX-License-Identifier: MIT
[CmdletBinding()]
param(
    [string]$BuildDirectory,
    [string]$RuntimeDirectory,
    [string]$BuildConfiguration = 'Release',
    [string]$Version = '0.1.0',
    [string]$DriverManifests = '',
    [string]$RequiredDriverManifests = ''
)
$ErrorActionPreference = 'Stop'
if (![Environment]::Is64BitOperatingSystem) { throw 'Parallel-Mater requires 64-bit Windows.' }
$sourceRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
if (!$BuildDirectory) { $BuildDirectory = Join-Path $sourceRoot 'build\Release' }
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$installRoot = [IO.Path]::GetFullPath((Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Parallel-Mater'))
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Parallel-Mater.lnk'
$registryPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Parallel-Mater'
$manifestPath = Join-Path $installRoot 'installation.json'
function Get-GalleryVersion([string]$Path) {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $Path
    $start.Arguments = '--version'
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($start)
    $output = $process.StandardOutput.ReadToEnd().Trim()
    $diagnostic = $process.StandardError.ReadToEnd().Trim()
    $process.WaitForExit()
    if ($process.ExitCode -ne 0 -or !$output) {
        throw "Could not identify gallery build '$Path' (exit $($process.ExitCode)): $diagnostic"
    }
    return $output
}
function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','')
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}
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
$running = Get-Process -Name 'Parallel-Mater' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and [IO.Path]::GetDirectoryName($_.Path) -eq $installRoot }
if ($running) { throw 'Close the installed Parallel-Mater gallery before updating it.' }
if (Test-Path -LiteralPath $shortcutPath) {
    $existing = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
    if ($existing.TargetPath -ne (Join-Path $installRoot 'Parallel-Mater.exe')) {
        throw "An unrelated Start menu shortcut already exists: $shortcutPath"
    }
}
$files = [ordered]@{}
$galleryExecutable = @(
    (Join-Path $BuildDirectory 'parallel-mater-gallery.exe'),
    (Join-Path $BuildDirectory 'Parallel-Mater.exe')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (!$galleryExecutable) {
    throw "Missing gallery executable in build directory: $BuildDirectory"
}
$files['Parallel-Mater.exe'] = (Resolve-Path -LiteralPath $galleryExecutable).Path
foreach ($dll in Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.dll' -File |
    Where-Object { $_.Name -ne 'parallel-mater-mock-driver.dll' }) {
    $files[$dll.Name] = $dll.FullName
}
if ($RuntimeDirectory) {
    foreach ($dll in Get-ChildItem -LiteralPath $RuntimeDirectory -Filter '*.dll' -File |
        Where-Object { $_.Name -ne 'parallel-mater-mock-driver.dll' }) {
        $files[$dll.Name] = $dll.FullName
    }
}
$packagedAssetDirectory = Join-Path $BuildDirectory 'assets'
if (!(Test-Path -LiteralPath $packagedAssetDirectory -PathType Container)) {
    throw "Missing packaged gallery assets: $packagedAssetDirectory"
}
$packagedAssetCount = 0
foreach ($asset in Get-ChildItem -LiteralPath $packagedAssetDirectory -Filter '*.glb' -File) {
    $files["assets\$($asset.Name)"] = $asset.FullName
    $packagedAssetCount++
}
if ($packagedAssetCount -eq 0) {
    throw "No packaged gallery scenes were found in: $packagedAssetDirectory"
}
$files['config\device-profiles.json'] = (Resolve-Path -LiteralPath (Join-Path $BuildDirectory 'config\device-profiles.json')).Path
$knownDriverManifests = @('slang-cuda.json','slang-d3d12.json','slang-cpu.json')
$selectedDriverManifests = if ($DriverManifests) {
    @($DriverManifests.Split(',') | Where-Object { $_ })
} else {
    @($knownDriverManifests | Where-Object {
        Test-Path -LiteralPath (Join-Path $BuildDirectory "manifests\portable\$_") -PathType Leaf
    })
}
foreach ($manifestName in $selectedDriverManifests) {
    if ($knownDriverManifests -notcontains $manifestName) {
        throw "Unknown Parallel-Mater driver manifest requested: $manifestName"
    }
    $driverManifest = Join-Path $BuildDirectory "manifests\portable\$manifestName"
    if (!(Test-Path -LiteralPath $driverManifest -PathType Leaf)) {
        throw "Missing configured Parallel-Mater driver manifest: $driverManifest"
    }
    $files["manifests\$manifestName"] = (Resolve-Path -LiteralPath $driverManifest).Path
}
if ($selectedDriverManifests.Count -eq 0) {
    throw "No portable Parallel-Mater driver manifest was found in: $BuildDirectory"
}
$requiredDriverManifestNames = @($RequiredDriverManifests.Split(',') | Where-Object { $_ })
foreach ($manifestName in $requiredDriverManifestNames) {
    if ($knownDriverManifests -notcontains $manifestName) {
        throw "Unknown required Parallel-Mater driver manifest: $manifestName"
    }
    if ($selectedDriverManifests -notcontains $manifestName) {
        throw "Canonical Parallel-Mater installation requires driver manifest: $manifestName"
    }
}
$files['manifests\validation.json'] = (Resolve-Path -LiteralPath (Join-Path $BuildDirectory 'manifests\portable\validation.json')).Path
$files['LICENSE'] = "$sourceRoot\LICENSE"
$files['GALLERY.md'] = "$sourceRoot\docs\GALLERY.md"
$files['Uninstall.ps1'] = "$PSScriptRoot\Uninstall-WindowsGallery.ps1"
# Resolve every input before making installation changes.
foreach ($source in $files.Values) { if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing package file: $source" } }
$reportedVersion = Get-GalleryVersion $galleryExecutable
$revision = 'unknown'
try {
    $candidateRevision = (& git -C $sourceRoot rev-parse HEAD 2>$null).Trim()
    if ($LASTEXITCODE -eq 0 -and $candidateRevision) { $revision = $candidateRevision }
} catch {}
$sourceHashes = [ordered]@{}
foreach ($name in $files.Keys) {
    $sourceHashes[$name] = Get-Sha256 $files[$name]
}
$obsoleteTargets = @($previousFiles | Where-Object {
    !$files.Contains($_)
} | ForEach-Object {
    $target = [IO.Path]::GetFullPath((Join-Path $installRoot $_))
    if (!$target.StartsWith($installRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe previous installation path: $_"
    }
    $target
})
$legacyTestDriver = Join-Path $installRoot 'parallel-mater-mock-driver.dll'
if ((Test-Path -LiteralPath $legacyTestDriver -PathType Leaf) -and
    $obsoleteTargets -notcontains $legacyTestDriver) {
    $obsoleteTargets += $legacyTestDriver
}
foreach ($directory in @($installRoot,'assets','config','manifests')) {
    $path = if ($directory -eq $installRoot) { $installRoot } else { Join-Path $installRoot $directory }
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}
$manifestFilesDuringInstall = @($previousFiles + @($files.Keys) | Sort-Object -Unique)
$manifest = [ordered]@{ application='Parallel-Mater'; directory=$installRoot; version=$Version;
    configuration=$BuildConfiguration; revision=$revision; reportedVersion=$reportedVersion;
    executableSha256=$sourceHashes['Parallel-Mater.exe']; state='installing';
    driverManifests=@($selectedDriverManifests); files=$manifestFilesDuringInstall;
    hashes=$sourceHashes }
# Keep the ownership marker even if a copy fails, so rerunning can repair it.
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
foreach ($name in $files.Keys) {
    $target = Join-Path $installRoot $name
    $temporary = Join-Path ([IO.Path]::GetDirectoryName($target)) ('.' + [IO.Path]::GetFileName($target) + ".installing.$PID")
    $backup = Join-Path ([IO.Path]::GetDirectoryName($target)) ('.' + [IO.Path]::GetFileName($target) + ".replacing.$PID")
    try {
        Copy-Item -LiteralPath $files[$name] -Destination $temporary -Force
        $copiedHash = Get-Sha256 $temporary
        if ($copiedHash -ne $sourceHashes[$name]) {
            throw "Package verification failed while copying: $name"
        }
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            [IO.File]::Replace($temporary,$target,$backup,$true)
            Remove-Item -LiteralPath $backup -Force
        } else {
            [IO.File]::Move($temporary,$target)
        }
    } finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}
foreach ($name in $files.Keys) {
    $installedHash = Get-Sha256 (Join-Path $installRoot $name)
    if ($installedHash -ne $sourceHashes[$name]) {
        throw "Installed package verification failed: $name"
    }
}
$installedReportedVersion = Get-GalleryVersion (Join-Path $installRoot 'Parallel-Mater.exe')
if ($installedReportedVersion -ne $reportedVersion) {
    throw "Installed gallery identity mismatch: expected '$reportedVersion', got '$installedReportedVersion'"
}
foreach ($target in $obsoleteTargets) {
    if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force }
}
foreach ($legacyName in @('drivers.d','layers.d')) {
    $legacyDirectory = Join-Path $installRoot $legacyName
    if ((Test-Path -LiteralPath $legacyDirectory -PathType Container) -and
        !(Get-ChildItem -LiteralPath $legacyDirectory -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $legacyDirectory -Force
    }
}
$manifest['state'] = 'installed'
$manifest['installedUtc'] = [DateTime]::UtcNow.ToString('o')
$manifest['files'] = @($files.Keys | Sort-Object -Unique)
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $installRoot 'Parallel-Mater.exe'
$shortcut.WorkingDirectory = $installRoot
$shortcut.Description = 'Parallel-Mater - Slang GPU Physics Gallery'
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
Write-Output "Build: $reportedVersion"
Write-Output "Start menu: $shortcutPath"
