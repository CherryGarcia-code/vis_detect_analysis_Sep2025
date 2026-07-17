<#
.SYNOPSIS
    PreToolUse guard: block recursive deletes that would follow a Windows
    directory JUNCTION out of the intended target and into the primary checkout.

.DESCRIPTION
    On 2026-06-07 this project lost ~45 session .pkl files and all of FIGURES/
    because `git worktree remove` was run on a worktree whose data/ and FIGURES/
    subdirectories were directory junctions pointing at the PRIMARY checkout.
    Git followed the junctions and deleted the targets. It did not need --force.

    DESIGN (deliberately not a regex blocklist):

      * A regex on the command text alone is both over- and under-inclusive.
        It blocks safe ops (a worktree with no junctions; `rm -rf data/cache/foo`)
        and it is trivially evaded (deny `git worktree remove <wt>` and the
        natural workaround is `rm -rf .claude/worktrees/<wt>` -- same blast
        radius, different words). A text-only guard creates a gradient toward
        its own blind spot.

      * So: match the destructive VERB broadly (cheap, over-inclusive on purpose),
        then decide by ACTUALLY LOOKING at the filesystem for reparse points.
        Deny only on a CONFIRMED junction; otherwise say nothing.

    Never emits "allow" -- failing to prove a command dangerous is not evidence
    that it is safe. Silence hands the command back to the normal permission flow.

.NOTES
    Input : PreToolUse hook payload (JSON) on stdin.
    Output: nothing (exit 0), or a PreToolUse deny decision as JSON on stdout.
#>

Set-StrictMode -Off
$ErrorActionPreference = 'Stop'

# --- tunables ---------------------------------------------------------------
$MAX_DEPTH       = 4      # levels below each candidate to inspect
$MAX_DIRS        = 20000  # hard cap on directories visited (time bound)
$MAX_CANDIDATES  = 40     # hard cap on path tokens we resolve
# Never DESCEND into these (their LinkType is still checked before we skip).
$SKIP_DESCENT    = @('.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache', '.mypy_cache')

# Destructive verbs. Broad on purpose -- a false match costs one bounded scan.
$DESTRUCTIVE = @(
    'git\s+worktree\s+remove',            # the exact 2026-06-07 incident
    '\brm\b[^\r\n]*?(--recursive|-{1,2}[a-zA-Z]*[rR])',   # rm -rf / -fr / -Rf / --recursive
    'Remove-Item',                        # any word order, -Recurse may precede -Path
    '(?<![\w-])(ri|rd|rmdir|del|erase)(?![\w-])',         # PS/cmd aliases
    'shutil\.rmtree', '\brmtree\b',       # python
    'os\.removedirs'
) -join '|'

function Write-Deny([string]$reason) {
    $payload = @{
        hookSpecificOutput = @{
            hookEventName           = 'PreToolUse'
            permissionDecision      = 'deny'
            permissionDecisionReason = $reason
        }
    }
    # -Compress keeps it to one line; Claude Code parses stdout as JSON.
    [Console]::Out.Write(($payload | ConvertTo-Json -Compress -Depth 5))
    exit 0
}

# --- 1. read the payload ----------------------------------------------------
try {
    $raw = [Console]::In.ReadToEnd()
    if ([string]::IsNullOrWhiteSpace($raw)) { exit 0 }
    $payload = $raw | ConvertFrom-Json
    $cmd = [string]$payload.tool_input.command
} catch {
    exit 0   # unparseable payload -> stay quiet, normal permission flow applies
}
if ([string]::IsNullOrWhiteSpace($cmd)) { exit 0 }

# --- 2. destructive verb? ---------------------------------------------------
if ($cmd -notmatch $DESTRUCTIVE) { exit 0 }

# --- 3. candidate path tokens ----------------------------------------------
# Everything that could name a directory, plus '.' (catches `cd data && rm -rf *`,
# where the deleted tree is the cwd and never appears as an argument).
function Get-Candidates([string]$command) {
    $out = [System.Collections.Generic.List[string]]::new()
    $out.Add('.') | Out-Null

    foreach ($m in [regex]::Matches($command, '[^\s;|&(){}]+')) {
        $t = $m.Value.Trim("'", '"', ',', '`')
        if (-not $t) { continue }
        if ($t.StartsWith('-')) { continue }            # a flag, not a path
        if ($t -match '^\w+://') { continue }           # a URL
        if ($t -eq '*' -or $t -eq './*') { $out.Add('.') | Out-Null; continue }

        # `rm -rf data/*` -> the thing at risk is `data`
        $t = $t -replace '[\\/]\*+$', ''
        if (-not $t) { continue }

        # MSYS/Git-Bash drive syntax: /e/foo/bar -> E:\foo\bar
        if ($t -match '^/([a-zA-Z])/(.*)$') { $t = "$($Matches[1]):\$($Matches[2])" }

        $out.Add($t) | Out-Null
    }
    return ($out | Select-Object -Unique | Select-Object -First $MAX_CANDIDATES)
}

function Resolve-Dir([string]$token) {
    try {
        $p = if ([System.IO.Path]::IsPathRooted($token)) { $token }
             else { Join-Path (Get-Location).Path $token }
        $p = [System.IO.Path]::GetFullPath($p)
        if ([System.IO.Directory]::Exists($p)) { return $p }
    } catch { }
    return $null
}

# --- 4. bounded junction scan ----------------------------------------------
# Iterative BFS. Records reparse points and NEVER descends through one, so the
# scan can't be dragged across a junction into the primary's multi-GB data/.
$script:visited = 0
function Find-Junctions([string]$root) {
    $hits = [System.Collections.Generic.List[string]]::new()

    try {
        $self = Get-Item -LiteralPath $root -Force -ErrorAction Stop
        if ($self.LinkType) {
            $hits.Add("$($self.FullName)  [$($self.LinkType)] -> $($self.Target)") | Out-Null
            return $hits   # the candidate IS a link; nothing below it is ours
        }
    } catch { return $hits }

    $queue = [System.Collections.Generic.Queue[object]]::new()
    $queue.Enqueue(@{ Path = $root; Depth = 0 }) | Out-Null

    while ($queue.Count -gt 0) {
        $node = $queue.Dequeue()
        if ($node.Depth -ge $MAX_DEPTH) { continue }
        if ($script:visited -ge $MAX_DIRS) { break }

        $children = @()
        try {
            $children = Get-ChildItem -LiteralPath $node.Path -Force -Directory -ErrorAction Stop
        } catch { continue }

        foreach ($c in $children) {
            $script:visited++
            if ($script:visited -ge $MAX_DIRS) { break }

            if ($c.LinkType) {
                # A reparse point. Record it; do NOT descend through it.
                $hits.Add("$($c.FullName)  [$($c.LinkType)] -> $($c.Target)") | Out-Null
                continue
            }
            if ($SKIP_DESCENT -contains $c.Name) { continue }
            $queue.Enqueue(@{ Path = $c.FullName; Depth = $node.Depth + 1 }) | Out-Null
        }
    }
    return $hits
}

$all = [System.Collections.Generic.List[string]]::new()
foreach ($tok in (Get-Candidates $cmd)) {
    $dir = Resolve-Dir $tok
    if (-not $dir) { continue }
    foreach ($h in (Find-Junctions $dir)) {
        if (-not $all.Contains($h)) { $all.Add($h) | Out-Null }
    }
}

# --- 5. decide --------------------------------------------------------------
if ($all.Count -eq 0) { exit 0 }   # no junction found -> SILENT. Never auto-allow.

$linkPaths = $all | ForEach-Object { ($_ -split '\s+\[')[0] }
$deleteCmds = $linkPaths | ForEach-Object {
    "  [System.IO.Directory]::Delete('$_', `$false)   # deletes the LINK only, leaves the target intact"
}

$reason = @"
BLOCKED: live directory JUNCTION(s) inside the delete target. A recursive delete
here can follow the junction and destroy the PRIMARY checkout's real data --
this is exactly the 2026-06-07 data loss (~45 session .pkl files + all of FIGURES/,
lost to a plain 'git worktree remove' -- no --force needed).

Command:
  $cmd

Junctions found:
$($all -join "`n")

REMEDY -- delete the LINKS first (this does NOT touch the targets):
$($deleteCmds -join "`n")

Then confirm none remain before you retry the delete:
  Get-ChildItem <target> -Recurse -Directory -Force | ? LinkType

NOTE: never `Remove-Item -Recurse` a junction -- that recurses into the target.
[System.IO.Directory]::Delete(<path>, `$false) removes only the reparse point.
"@

Write-Deny $reason
