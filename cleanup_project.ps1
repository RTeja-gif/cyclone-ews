# cleanup_project.ps1
# Safely clean up unneeded files for cyclone-ews project by MOVING them into a backup folder.
# Run this from the project root (C:\Users\Teja\OneDrive\Desktop\cyclone-ews)

$now = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path -Path "." -ChildPath ("cleanup_backup_$now")
New-Item -ItemType Directory -Path $backup | Out-Null

Write-Host "Backup folder:" $backup

# Files and patterns to move (adjust if any path not present)
$targets = @(
    # old dashboards
    "dashboard\app.py",
    "dashboard\app_v2.py",
    "dashboard\app_v4.py",

    # old src scripts (examples from conversation) - adjust or add if other names exist
    "src\create_sequences.py",
    "src\create_sequences_force_features.py",
    "src\create_sequences_force_features.py.backup",
    "src\create_sequences.py.backup",
    "src\create_sequences_force_features.py",
    "src\create_sequences.py",
    "src\create_sequences_force_features.py",
    "src\create_sequences.py",
    "src\create_sequences_force_features.py",
    "src\evaluate.py",
    "src\inspect_data.py",
    "src\inspect_press_wind.py",
    "src\lstm_model.py",
    "src\predict_one.py",
    "src\test_data.py",
    "src\train_lstm.py",
    "src\train_wind_model.py",
    "src\create_sequences.py",
    "src\create_sequences_force_features.py",
    "src\evaluate.py",

    # old models (optional) - move if present
    "models\lstm_best.h5",
    "models\lstm_final.h5"
)

# Also include any files matching certain patterns (older versions)
$patterns = @(
    "dashboard\app_*.py",
    "src\*old*.py",
    "src\*test*.py",
    "src\*legacy*.py"
)

# Move each explicit target if it exists
foreach ($t in $targets) {
    if (Test-Path $t) {
        $dest = Join-Path $backup (Split-Path $t -Leaf)
        Write-Host "Moving" $t "->" $dest
        Move-Item -Path $t -Destination $dest -Force
    }
    else {
        Write-Host "Not found:" $t
    }
}

# Move files found by glob patterns
foreach ($p in $patterns) {
    $found = Get-ChildItem -Path $p -File -ErrorAction SilentlyContinue
    foreach ($f in $found) {
        $dest = Join-Path $backup $f.Name
        Write-Host "Moving" $($f.FullName) "->" $dest
        Move-Item -Path $f.FullName -Destination $dest -Force
    }
}

# Additional cleanup: remove empty directories (only if empty)
$dirsToCheck = @("dashboard","src","models")
foreach ($d in $dirsToCheck) {
    if (Test-Path $d) {
        $items = Get-ChildItem -Path $d -Force | Where-Object { -not ($_.PSIsContainer) } 
        if ($items.Count -eq 0) {
            Write-Host "Directory" $d "contains no files (non-recursive). Keeping directory."
        } else {
            Write-Host "Directory" $d "has" $items.Count "files (non-recursive)."
        }
    }
}

Write-Host ""
Write-Host "=== Done. Check backup folder for moved files ==="
Get-ChildItem -Path $backup | ForEach-Object { Write-Host $_.FullName }
