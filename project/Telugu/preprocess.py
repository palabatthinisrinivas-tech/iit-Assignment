# preprocess.py : (1) write config/<lang>/*.yaml  (2) MFA forced alignment  (3) FastSpeech2 features  (4) exact train/val split
import os, sys, json, glob, argparse, subprocess, yaml
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
cfg = yaml.safe_load(open('config.yaml')); LANG = cfg['lang']; P = cfg['preprocess']
RAW = P['path']['raw_path']; OUT = P['path']['preprocessed_path']; TG = os.path.join(OUT, 'TextGrid')
MFA = ('MAMBA_ROOT_PREFIX=/kaggle/temp/mamba/root MFA_ROOT_DIR=/kaggle/temp/mfa_root '
       '/kaggle/temp/mamba/bin/micromamba run -n aligner mfa')

def run(cmd):
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end='', flush=True)
    if p.wait(): sys.exit('command failed: ' + cmd)

def split_config():
    d = os.path.join('config', LANG); os.makedirs(d, exist_ok=True)
    for k in ['preprocess', 'model', 'train']:
        yaml.safe_dump(cfg[k], open(os.path.join(d, k + '.yaml'), 'w'), allow_unicode=True)
    print('wrote', d)

def align():
    if glob.glob(os.path.join(TG, '*', '*.TextGrid')):
        print('TextGrids already exist -> skipping alignment'); return
    os.makedirs(OUT, exist_ok=True)
    if LANG == 'English':      # pretrained English ARPA acoustic model + dictionary
        run(MFA + ' model download acoustic english_us_arpa || true')
        run(MFA + ' model download dictionary english_us_arpa || true')
        run(f'{MFA} align --clean -j 4 --beam 100 --retry_beam 400 {RAW} english_us_arpa english_us_arpa {TG}')
    else:                      # Telugu: no pretrained model -> grapheme dictionary + acoustic model trained on our 6 h
        words = set()
        for f in glob.glob(RAW + '/*/*.lab'): words.update(open(f, encoding='utf8').read().split())
        dic = os.path.join(OUT, 'telugu_graphemes.dict')
        with open(dic, 'w', encoding='utf8') as f:
            for w in sorted(words): f.write(w + '\t' + ' '.join(list(w)) + '\n')
        print('dictionary words:', len(words))
        open(OUT + '/mfa_train.yaml', 'w').write('beam: 10\nretry_beam: 40\ntraining:\n  - monophone:\n      num_iterations: 20\n      max_gaussians: 1000\n      subset: 0\n')
        run(f'{MFA} train --clean -j 4 --config_path {OUT}/mfa_train.yaml {RAW} {dic} {OUT}/telugu_mfa_acoustic.zip --output_directory {TG}')

def features():
    from preprocessor.preprocessor import Preprocessor
    Preprocessor(P).build_from_path()

def resplit():
    sp = json.load(open(cfg['split_json'])); tr, va = set(sp['train']), set(sp['val'])
    assert not tr & va
    lines = []
    for fn in ['train.txt', 'val.txt']:
        lines += [l.rstrip('\n') for l in open(os.path.join(OUT, fn), encoding='utf8') if l.strip()]
    sel = {'train': [l for l in lines if l.split('|')[0] in tr], 'val': [l for l in lines if l.split('|')[0] in va]}
    hop, sr = P['preprocessing']['stft']['hop_length'], P['preprocessing']['audio']['sampling_rate']
    for k in sel:
        with open(os.path.join(OUT, k + '.txt'), 'w', encoding='utf8') as f: f.write('\n'.join(sel[k]) + '\n')
        fr = sum(np.load(os.path.join(OUT, 'mel', '%s-mel-%s.npy' % (l.split('|')[1], l.split('|')[0])), mmap_mode='r').shape[0] for l in sel[k])
        print('%s: %d utts (of %d selected), %.2f h after alignment/trimming' % (k, len(sel[k]), len(sp[k]), fr * hop / sr / 3600))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='all', choices=['all', 'split_config', 'align', 'features', 'resplit'])
    a = ap.parse_args(); split_config()
    if a.stage in ('all', 'align'): align()
    if a.stage in ('all', 'features'): features()
    if a.stage in ('all', 'resplit'): resplit()
