**WARNING:** The Python implementation of Prefix is deprecated. Developers should focus on bringing the [C implementation of Prefix](https://github.com/python-processing-unit/Prefix-C) to full specification compatability.

## Installation
<code>Invoke-WebRequest -Uri "https://github.com/python-processing-unit/Prefix/archive/refs/heads/main.zip" -OutFile "path\to\download\Prefix.zip"<br>
Expand-Archive -Path "path\to\download\Prefix.zip" -DestinationPath "path\to\extract\Prefix"<br>
$old = [Environment]::GetEnvironmentVariable('Path','User')<br>
if(-not $old.Split(';') -contains 'path\to\extract\Prefix'){ [Environment]::SetEnvironmentVariable('Path',$old + ';path\to\extract\Prefix','User') }<br>
Remove-Item -Path "path\to\download\Prefix.zip"</code>
