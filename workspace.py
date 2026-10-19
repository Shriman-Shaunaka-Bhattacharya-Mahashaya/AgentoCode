"""Shared inventory and tool boundary. Project input is never executed here."""
from pathlib import Path
import hashlib
import os
import subprocess

APP_ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.git', '.hg', '.svn', '.venv', 'venv', 'node_modules', '__pycache__',
            'knowledge_base', '.guardian', '.chroma_db', 'dist', 'build', '.pytest_cache'}
SECRET_NAMES = {'.env', '.cg_history', 'credentials', 'credentials.json', 'id_rsa',
                'id_ed25519', '.netrc'}

def digest(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()

def safe_path(root, relative):
    root = Path(root).resolve(strict=True)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('Path is outside the selected workspace')
    # Reject links/junctions even if they currently point inside the workspace.
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Links and junctions are not supported')
    original = root / relative
    for parent in [original, *original.parents]:
        if parent == root:
            break
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError('Links and junctions are not supported')
    return candidate

def validate_agent_workspace(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or APP_ROOT.is_relative_to(root):
        raise ValueError('Select a project workspace that does not contain the agent application')
    return root

def excluded(relative):
    parts = Path(relative).parts
    return any(p in EXCLUDED or p.startswith('.') or p.lower() in SECRET_NAMES or
               p.lower().endswith(('.pem', '.key', '.p12', '.pfx')) for p in parts)

def inventory(root, max_bytes=512_000):
    """Same eligibility rules for initial scans, reconciliation, and tool discovery."""
    root = Path(root).resolve()
    files, directories, skipped = {}, {''}, {}
    candidates = []
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not excluded((Path(base)/d).relative_to(root))
                         and not (Path(base)/d).is_symlink()
                         and not (hasattr(Path(base)/d, 'is_junction') and (Path(base)/d).is_junction()))
        for name in dirs + sorted(names):
            p = Path(base) / name
            rel = p.relative_to(root).as_posix()
            if not excluded(rel):
                candidates.append(rel + ('/' if name in dirs else ''))
    ignored = set()
    git_checked = False
    # Git provides exact nested/negated ignore semantics without executing project code.
    if candidates:
        try:
            result = subprocess.run(['git', '-C', str(root), 'check-ignore', '--stdin', '-z'],
                                    input='\0'.join(candidates)+'\0', capture_output=True,
                                    text=True, timeout=10)
            if result.returncode in (0, 1):
                ignored = set(result.stdout.split('\0'))
                git_checked = True
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not git_checked:
        # Non-Git workspaces still honor their local and nested .gitignore files.
        from pathspec import GitIgnoreSpec
        specs = {}
        for rel in candidates:
            path = root / rel.rstrip('/')
            ignored_here = False
            # Use ordered source ancestors, with later negated rules overriding earlier ones.
            ancestors = [root]
            relative_parent = path.parent.relative_to(root)
            for length in range(1,len(relative_parent.parts)+1):
                ancestors.append(root.joinpath(*relative_parent.parts[:length]))
            for parent in ancestors:
                ignorefile = parent / '.gitignore'
                if ignorefile not in specs:
                    specs[ignorefile] = GitIgnoreSpec.from_lines(ignorefile.read_text(encoding='utf-8').splitlines()) if ignorefile.is_file() else None
                spec = specs[ignorefile]
                if spec:
                    match = spec.check_file(path.relative_to(parent).as_posix() + ('/' if rel.endswith('/') else ''))
                    if match.include is not None:
                        ignored_here = match.include
            if ignored_here:
                ignored.add(rel)
    for rel in candidates:
        if rel in ignored:
            continue
        try:
            p = safe_path(root, rel.rstrip('/'))
            if rel.endswith('/'):
                directories.add(rel.rstrip('/'))
                continue
            if p.stat().st_size > max_bytes:
                skipped[rel] = 'File exceeds configured byte limit'
                continue
            raw = p.read_bytes()
            if b'\0' in raw:
                skipped[rel] = 'Binary input'
                continue
            source = raw.decode('utf-8')
            files[rel] = {'source': source, 'hash': digest(raw)}
        except (OSError, UnicodeError, ValueError) as exc:
            skipped[rel] = type(exc).__name__
    return files, directories, skipped
