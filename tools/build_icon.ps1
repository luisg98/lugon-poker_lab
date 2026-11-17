$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$srcPath = "C:\Users\luisg\.gemini\antigravity\brain\176dc29d-8909-473e-b069-cf4cd03a2ce2\lugon_poker_lab_logo_1794846159351.jpg"
$assetsDir = "c:\Users\luisg\Desktop\lugon-poker_lab\assets"
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

$pngPath = Join-Path $assetsDir "lugon_poker_lab.png"
$png64Path = Join-Path $assetsDir "lugon_poker_lab_64.png"
$icoPath = Join-Path $assetsDir "lugon_poker_lab.ico"

$srcImg = [System.Drawing.Image]::FromFile($srcPath)
$srcImg.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)

$sizes = @(16, 32, 48, 64, 256)
$pngStreams = New-Object System.Collections.Generic.List[byte[]]

foreach ($sz in $sizes) {
    $bmp = New-Object System.Drawing.Bitmap($sz, $sz)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.DrawImage($srcImg, 0, 0, $sz, $sz)
    $g.Dispose()

    if ($sz -eq 64) {
        $bmp.Save($png64Path, [System.Drawing.Imaging.ImageFormat]::Png)
    }

    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $pngStreams.Add($ms.ToArray())
    $ms.Dispose()
    $bmp.Dispose()
}
$srcImg.Dispose()

# Write multi-resolution ICO file
$fs = [System.IO.File]::Create($icoPath)
$bw = New-Object System.IO.BinaryWriter($fs)

# ICONDIR header (6 bytes)
$bw.Write([uint16]0)
$bw.Write([uint16]1)
$bw.Write([uint16]$sizes.Count)

$offset = 6 + (16 * $sizes.Count)
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $sz = $sizes[$i]
    $data = $pngStreams[$i]
    $wByte = if ($sz -ge 256) { [byte]0 } else { [byte]$sz }
    $hByte = if ($sz -ge 256) { [byte]0 } else { [byte]$sz }
    $bw.Write($wByte)
    $bw.Write($hByte)
    $bw.Write([byte]0)
    $bw.Write([byte]0)
    $bw.Write([uint16]1)
    $bw.Write([uint16]32)
    $bw.Write([uint32]$data.Length)
    $bw.Write([uint32]$offset)
    $offset += $data.Length
}

for ($i = 0; $i -lt $sizes.Count; $i++) {
    $bw.Write($pngStreams[$i])
}

$bw.Flush()
$bw.Close()
$fs.Close()

# Compile LugonPokerLab.exe with the embedded Win32 icon
Set-Location "c:\Users\luisg\Desktop\lugon-poker_lab"
& "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:winexe /optimize+ /win32icon:"assets\lugon_poker_lab.ico" /out:LugonPokerLab.exe tools\launcher.cs

# Refresh Windows Explorer icon cache
$code = @'
using System;
using System.Runtime.InteropServices;
public class ShellNotify {
    [DllImport("shell32.dll")]
    public static extern void SHChangeNotify(int wEventId, uint uFlags, IntPtr dwItem1, IntPtr dwItem2);
    public static void Refresh() {
        SHChangeNotify(0x08000000, 0x0000, IntPtr.Zero, IntPtr.Zero);
    }
}
'@
Add-Type -TypeDefinition $code
[ShellNotify]::Refresh()
Write-Output "Icon and LugonPokerLab.exe built successfully."
