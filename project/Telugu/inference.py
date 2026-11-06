# inference.py : synthesize the test sentences, plot train/val losses and variance-adaptor outputs
# (duration / pitch / energy) for ONE language.   usage: python inference.py --out <Results/Language> [--restore_step N]
import os, re, sys, csv, json, glob, argparse, unicodedata, string
import yaml, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.io import wavfile
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from utils.model import get_model, get_vocoder, vocoder_infer
from text import text_to_sequence
from text.symbols import symbols

cfg = yaml.safe_load(open('config.yaml')); LANG = cfg['lang']
PRE, MOD, TRN = cfg['preprocess'], cfg['model'], cfg['train']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR = PRE['preprocessing']['audio']['sampling_rate']; HOP = PRE['preprocessing']['stft']['hop_length']

TESTS = {
 'English': {
  'E1': 'The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.',
  'E2a': "The orchestra's conduct during the ceremony was praised by the audience.",
  'E2b': 'The researchers will conduct an experiment to evaluate the proposed method.',
  'E3': 'The rhyth m of the orchestra changed unexpectedly.',
  'E4': 'The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.'},
 'Telugu': {
  'T1': 'న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.',
  'T2a': 'తాడుపై అతని పట్టుచాలా బలంగా ఉంది.',
  'T2b': 'ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.',
  'T3': 'పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.',
  'T4': 'ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.'}}[LANG]

def english_sequence(text):
    from g2p_en import G2p                      # POS-aware G2P: handles homographs (conduct n./v.), numbers and OOV words
    toks = []
    for p in G2p()(text.rstrip(string.punctuation + ' ')):
        if p == ' ': continue
        toks.append('sp' if p in string.punctuation else p)
    return text_to_sequence('{' + ' '.join(toks) + '}', ['english_cleaners'])

def telugu_sequence(text):                       # grapheme (Unicode code point) tokens, same units used for MFA training
    text = unicodedata.normalize('NFC', text).replace('\u200c', '').replace('\u200d', '').rstrip(' .?!,;:')
    toks = []
    for ch in text:
        if ch.isspace(): continue
        if ch in ',;:.?!': toks.append('sp')
        elif '\u0c00' <= ch <= '\u0c7f' or ch in '0123456789': toks.append(ch)
    return text_to_sequence('{' + ' '.join(toks) + '}', ['english_cleaners'])

def latest_step():
    return max(int(os.path.basename(p).split('.')[0]) for p in glob.glob(os.path.join(TRN['path']['ckpt_path'], '*.pth.tar')))

def pick_font():
    font_paths = [
        '/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansTelugu-Regular.otf',
        '/usr/share/fonts/truetype/noto/NotoSansTelugu-Medium.ttf'
    ]
    for path in font_paths:
        if os.path.exists(path):
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            print('Successfully loaded Telugu font from:', path)
            return prop.get_name()
    print('WARNING: NotoSansTelugu font file not found, labels may show as boxes')
    return None

def bar_plot(vals, labels, title, ylabel, path, line=False):
    fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(vals)), 4))
    x = np.arange(len(vals))
    ax.plot(x, vals, marker='o') if line else ax.bar(x, vals)
    if labels is not None and len(labels) == len(vals):
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel); ax.set_xlabel('input token (phoneme / grapheme)')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_losses(out):
    keys = ['Total Loss', 'Mel Loss', 'Mel PostNet Loss', 'Pitch Loss', 'Energy Loss', 'Duration Loss']
    for split, fn in [('train', 'training_loss.png'), ('val', 'validation_loss.png')]:
        path = os.path.join(TRN['path']['log_path'], split, 'log.txt')
        if not os.path.exists(path): print('missing', path); continue
        rows = {}
        for line in open(path, encoding='utf8'):
            m = re.search(r'Step (\d+)', line)
            if not m or 'Total Loss' not in line: continue
            rows[int(m.group(1))] = [float(re.search(re.escape(k) + r': ([0-9.eE+-]+)', line).group(1)) for k in keys]
        st = sorted(rows); A = np.array([rows[s] for s in st])
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(st, A[:, 0]); ax[0].set_title(LANG + ' ' + split + ' total loss'); ax[0].set_xlabel('step'); ax[0].grid(alpha=.3)
        for j, k in enumerate(keys[1:], 1): ax[1].plot(st, A[:, j], label=k)
        ax[1].set_title(LANG + ' ' + split + ' loss components'); ax[1].set_xlabel('step'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, fn), dpi=150); plt.close(fig)
        print(fn, '| first/last total loss: %.3f -> %.3f | min %.3f at step %d' % (A[0, 0], A[-1, 0], A[:, 0].min(), st[int(A[:, 0].argmin())]))

@torch.no_grad()
def synth(model, vocoder, stats, tid, text, out):
    seq = np.array(english_sequence(text) if LANG == 'English' else telugu_sequence(text))
    labels = [symbols[i].lstrip('@') for i in seq]
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(dev)
    o = model(torch.tensor([0]).to(dev), texts, torch.tensor([len(seq)]).to(dev), len(seq))
    postnet, p, e, d = o[1], o[2][0].cpu().numpy(), o[3][0].cpu().numpy(), o[5][0].cpu().numpy()
    mel_len = int(o[9][0])
    wav = vocoder_infer(postnet[:, :mel_len].transpose(1, 2), vocoder, MOD, PRE, lengths=[mel_len * HOP])[0]
    wavfile.write(os.path.join(out, 'test_generated_audios', tid + '.wav'), SR, wav)
    if PRE['preprocessing']['pitch']['normalization']: p = p * stats['pitch'][3] + stats['pitch'][2]
    if PRE['preprocessing']['energy']['normalization']: e = e * stats['energy'][3] + stats['energy'][2]
    ms = d * HOP / SR * 1000.0
    vp = os.path.join(out, 'variance_adaptor_plots')
    ttl = '%s | %s' % (tid, text if len(text) < 75 else text[:72] + '...')
    bar_plot(ms, labels, 'Predicted duration - ' + ttl, 'duration (ms)', os.path.join(vp, 'duration', tid + '.png'))
    bar_plot(p, labels, 'Predicted pitch (F0) - ' + ttl, 'F0 (Hz)', os.path.join(vp, 'pitch', tid + '.png'), line=True)
    bar_plot(e, labels, 'Predicted energy - ' + ttl, 'energy (L2 norm of mel frame)', os.path.join(vp, 'energy', tid + '.png'))
    with open(os.path.join(vp, 'values_' + tid + '.csv'), 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f); w.writerow(['idx', 'token', 'duration_frames', 'duration_ms', 'pitch_hz', 'energy'])
        for i in range(len(seq)):
            if len(p) == len(seq): w.writerow([i, labels[i], int(d[i]), round(float(ms[i]), 1), round(float(p[i]), 2), round(float(e[i]), 3)])
    print('%-4s tokens=%3d frames=%4d dur=%.2fs' % (tid, len(seq), mel_len, mel_len * HOP / SR))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--restore_step', type=int, default=0)
    a = ap.parse_args(); step = a.restore_step or latest_step(); print(LANG, 'checkpoint step', step)
    for sub in ['test_generated_audios'] + ['variance_adaptor_plots/' + k for k in ('duration', 'pitch', 'energy')]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    json.dump(TESTS, open(os.path.join(a.out, 'test_sentences.json'), 'w'), ensure_ascii=False, indent=1)
    if LANG == 'Telugu': print('font:', pick_font())
    plot_losses(a.out)
    model = get_model(argparse.Namespace(restore_step=step), (PRE, MOD, TRN), dev, train=False)
    vocoder = get_vocoder(MOD, dev)
    stats = json.load(open(os.path.join(PRE['path']['preprocessed_path'], 'stats.json')))
    for tid, text in TESTS.items(): synth(model, vocoder, stats, tid, text, a.out)
