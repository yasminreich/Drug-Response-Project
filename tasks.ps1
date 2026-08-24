<#
.SYNOPSIS
    Windows equivalent of the Makefile targets.

.DESCRIPTION
    Same commands as the Makefile, for machines without `make` (the default on
    Windows). Everything runs inside the drug_response_env Docker image -- never
    on the host, per CLAUDE.md.

.EXAMPLE
    .\tasks.ps1 test
    .\tasks.ps1 notebook -Notebook 01_initial_EDA
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'build', 'test', 'lint', 'notebook', 'jupyter', 'clean')]
    [string]$Task = 'help',

    [string]$Notebook = '03_baselines_comparison'
)

$ErrorActionPreference = 'Stop'

$Image = 'drug_response_env'
$Repo = $PSScriptRoot
$Mount = @('-v', "${Repo}:/app", '-w', '/app')

function Invoke-InImage {
    # NB: do not name this parameter $Args -- that is a PowerShell automatic
    # variable, and binding it silently yields an empty array, so `docker run`
    # falls through to the image's default CMD (the blocking Jupyter server).
    param([string[]]$DockerArgs)
    & docker run --rm @Mount $Image @DockerArgs
    if ($LASTEXITCODE -ne 0) { throw "docker exited with $LASTEXITCODE" }
}

switch ($Task) {
    'help' {
        Write-Host 'Available tasks:'
        Write-Host '  build     Build the Docker image'
        Write-Host '  test      Run the pytest suite (no 518 MB matrix needed)'
        Write-Host '  lint      Run ruff over src/ and tests/'
        Write-Host '  notebook  Execute a notebook in place (-Notebook <name>)'
        Write-Host '  jupyter   Start Jupyter on http://localhost:8888'
        Write-Host '  clean     Remove generated artifacts from output/'
    }
    'build' {
        & docker build -t $Image $Repo
        if ($LASTEXITCODE -ne 0) { throw "docker build failed with $LASTEXITCODE" }
    }
    'test' { Invoke-InImage @('python', '-m', 'pytest', 'tests/') }
    'lint' { Invoke-InImage @('ruff', 'check', '.') }
    'notebook' {
        Invoke-InImage @(
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
            '--ExecutePreprocessor.timeout=7200', "notebooks/$Notebook.ipynb"
        )
    }
    'jupyter' {
        & docker run -it --rm @Mount -p 8888:8888 $Image
    }
    'clean' {
        Get-ChildItem -Path (Join-Path $Repo 'output') -Include *.png, *.csv, *.npy, *.npz, *.json `
            -File -ErrorAction SilentlyContinue | Remove-Item -Force
        Write-Host 'Cleaned output/ (it is gitignored anyway).'
    }
}
