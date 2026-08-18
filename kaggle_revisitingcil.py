# =============================================================================
# RevisitingCIL for IP102 (SimpleCIL / Aper) - KAGGLE LAUNCHER SCRIPT
#
# Cach dung:
#   - Kaggle Notebook: upload ca repo RevisitingCIL (zip, bo .git) lam private
#     dataset + add "IP102 dataset" (train.json/test.json/val.json +
#     VOC2007/VOC2007/JPEGImages) lam Input, roi paste file nay vao 1 cell:
#
#       !python /kaggle/input/<ten-code-slug>/kaggle_revisitingcil.py --model simplecil
#
#   - Hoac chay truc tiep:  python kaggle_revisitingcil.py --model simplecil
#
# Mo hinh: ViT-B/16 pretrained ImageNet-21k (timm tai ve tu dong, can Internet).
# Chia task: 7 / 6 / 6 / 6 (25 lop IP102).
# Ket qua ghi vao /kaggle/working/logs/...  (results.csv + history.json + *.log)
#
# Bien moi truong:
#   IP102_MODEL      - 'simplecil' (mac dinh) hoac 'aper_finetune'
#   IP102_MAX_TASKS  - chi chay N task dau (test nhanh), VD: 1 -> 1 task
#   IP102_DATA_ROOT  - duong dan thu muc dataset (neu khong muon auto-find)
#   IP102_MEMORY_SIZE- tong so exemplar (mac dinh 2000)
# =============================================================================

import json
import os
import shutil
import subprocess
import sys

DEFAULT_MODEL = os.environ.get('IP102_MODEL', 'simplecil').strip().lower()
MAX_TASKS = int(os.environ.get('IP102_MAX_TASKS', 0))
MEMORY_SIZE = int(os.environ.get('IP102_MEMORY_SIZE', 2000))
WORK_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()

MODEL_CONFIGS = {
    'simplecil': 'exps/simplecil/ip102_7_6_6_6_vit-b_simplecil.json',
    'aper_finetune': 'exps/aper_finetune/ip102_7_6_6_6_vit-b_finetune.json',
}

NUM_CLASSES = 25
INIT_CLS = 7


def find_code_dir():
    """Tim thu muc chua repo RevisitingCIL tren /kaggle/input."""
    for base in ['/kaggle/input']:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            cand = os.path.join(base, name)
            if not os.path.isdir(cand):
                continue
            if os.path.exists(os.path.join(cand, 'main.py')) and \
               os.path.exists(os.path.join(cand, 'trainer.py')):
                return cand
            for sub in sorted(os.listdir(cand)):
                subdir = os.path.join(cand, sub)
                if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, 'main.py')):
                    return subdir
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, 'main.py')):
        return here
    raise FileNotFoundError('Khong tim thay thu muc code RevisitingCIL (main.py)')


def find_data_root():
    if os.environ.get('IP102_DATA_ROOT'):
        root = os.environ['IP102_DATA_ROOT']
        if os.path.exists(os.path.join(root, 'train.json')):
            return root
    if os.path.isdir('/kaggle/input'):
        for dirpath, dirnames, filenames in os.walk('/kaggle/input'):
            if 'train.json' in filenames:
                return dirpath
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, 'IP102 dataset'),
                 os.path.dirname(os.path.dirname(here)) + '/IP102 dataset']:
        if os.path.exists(os.path.join(cand, 'train.json')):
            return cand
    raise FileNotFoundError('Khong tim thay thu muc IP102 dataset (train.json)')


def make_config(code_dir, model):
    src = os.path.join(code_dir, MODEL_CONFIGS[model])
    cfg = json.load(open(src, encoding='utf-8'))
    cfg['memory_size'] = MEMORY_SIZE
    if MAX_TASKS > 0:
        n_tasks = min(MAX_TASKS, NUM_CLASSES // INIT_CLS + 1)
        if n_tasks <= 1:
            cfg['increment'] = NUM_CLASSES - INIT_CLS
        else:
            cfg['increment'] = -(-(NUM_CLASSES - INIT_CLS) // (n_tasks - 1))
    dst = os.path.join(WORK_DIR, 'config_%s.json' % model)
    json.dump(cfg, open(dst, 'w', encoding='utf-8'), indent=2)
    return dst


def main():
    print('DATA_ROOT:', find_data_root())
    print('WORK_DIR :', WORK_DIR)
    code_dir = find_code_dir()
    print('CODE_DIR :', code_dir)

    if code_dir != WORK_DIR:
        work_code = os.path.join(WORK_DIR, 'RevisitingCIL')
        if not os.path.exists(work_code):
            print('Copying code to %s ...' % work_code)
            shutil.copytree(code_dir, work_code,
                            ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git'))
        code_dir = work_code

    models = [m.strip() for m in DEFAULT_MODEL.split(',') if m.strip()]
    for model in models:
        if model not in MODEL_CONFIGS:
            print('Bo qua model khong ho tro: %s (ho tro: %s)'
                  % (model, ', '.join(MODEL_CONFIGS)))
            continue
        cfg_path = make_config(code_dir, model)
        cmd = [sys.executable, 'main.py', '--config', cfg_path]
        print('==== running %s ====' % model)
        print(' '.join(cmd))
        ret = subprocess.call(cmd, cwd=code_dir)
        print('==== %s finished, return code %s ====' % (model, ret))

    print('Done. Ket qua trong: %s' % os.path.join(WORK_DIR, 'logs'))


if __name__ == '__main__':
    main()