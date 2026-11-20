"""Rebuild results/*.json from the raw batches in a temporary directory and compare with the shipped files."""
import glob, json, math, os, subprocess, sys, tempfile

root = os.path.dirname(os.path.abspath(__file__))


def same(a, b, path):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{path}[{i}]')
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def run(cmd, cwd):
    script, *args = cmd.split()
    subprocess.run([sys.executable, os.path.join(root, script), *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


with tempfile.TemporaryDirectory() as tmp:
    os.makedirs(os.path.join(tmp, 'results'))
    for f in glob.glob(os.path.join(root, 'results', '*_raw_*')):
        os.symlink(f, os.path.join(tmp, 'results', os.path.basename(f)))
    for cmd in open(os.path.join(root, 'pool_commands.txt')).read().split('\n'):
        if cmd:
            run(cmd, tmp)
    for name in ('flatness', 'finer_bins', 'quad_stats'):
        same(json.load(open(os.path.join(tmp, 'results', f'{name}.json'))), json.load(open(os.path.join(root, 'results', f'{name}.json'))), name)
        print(f'results/{name}.json: rebuilt from the raw batches, equal to the shipped file')
    run('gbs_hash_classical.py', tmp)                      # checks inside; timings differ between machines, the bits must not
    new, old = (json.load(open(os.path.join(d, 'results', 'gbs_hash.json'))) for d in (tmp, root))
    assert [(r['modes'], r['hash']) for r in new] == [(r['modes'], r['hash']) for r in old]
    print('results/gbs_hash.json: hash bits equal to the shipped file;', ', '.join(f"{r['modes']} modes in {r['seconds']:.2f} s" for r in new))
