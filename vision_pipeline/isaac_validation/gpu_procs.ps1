# Helper for gpu_guard.sh: list GPU-type processes that are not this check's
# own ancestors, or the caller's (args[0] = its PID): the job asking "is the
# GPU free?" must not count itself.
# GPU-type = Isaac (.venv-isaac / isaacsim / kit), SAM + CLIP (.venv-vision),
# the sim repo, or anything launched from a Claude session's scratchpad.
# CPU-only geometry (.venv-cad) never counts.
$all = @(Get-CimInstance Win32_Process)
$byId = @{}
foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }
$self = @{}
foreach ($start in @([int]$PID, [int]$args[0])) {   # this check, and the caller's PID if given
    $cur = $start
    while ($cur -gt 0 -and $byId.ContainsKey($cur) -and -not $self.ContainsKey($cur)) {
        $self[$cur] = $true
        $cur = [int]$byId[$cur].ParentProcessId
    }
}
foreach ($p in $all) {
    $cl = $p.CommandLine
    if (-not $cl -or $self.ContainsKey([int]$p.ProcessId)) { continue }
    if ($p.Name -notmatch 'python|kit') { continue }
    if ($cl -notmatch '(\.venv-isaac|\.venv-vision|isaacsim|biotech-lab-robotics|AppData\\Local\\Temp/claude/)') { continue }
    if ($cl -match '(\.venv-cad|Get-CimInstance|telemetry)') { continue }
    "$($p.ProcessId) $($cl.Substring(0, [Math]::Min(110, $cl.Length)))"
}
