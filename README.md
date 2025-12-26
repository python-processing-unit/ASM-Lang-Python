## Installation
`Invoke-WebRequest -Uri "https://github.com/python-processing-unit/ASM-Lang/archive/refs/heads/main.zip" -OutFile "path\to\download\ASM-Lang.zip"`<br>
`Expand-Archive -Path "path\to\download\ASM-Lang.zip" -DestinationPath "path\to\extract\ASM-Lang"`<br>
`$old = [Environment]::GetEnvironmentVariable('Path','User')`<br>
`if(-not $old.Split(';') -contains 'path\to\extract\ASM-Lang'){ [Environment]::SetEnvironmentVariable('Path',$old + ';path\to\extract\ASM-Lang','User') }`<br>
`Remove-Item -Path "path\to\download\ASM-Lang.zip"`<br>

## Crypto extension
The crypto extension (RSA, AES-GCM, AES-CBC) requires the Python package `cryptography`.

Install:
`python -m pip install cryptography`

Use:
`IMPORT(crypto)`

## Tests
Run the crypto + GC tests:
`python asm-lang.py test_crypto_gc.asmln`
