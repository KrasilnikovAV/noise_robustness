"""Run the complete AG News typo experiment with seed 42."""
import os
from pathlib import Path
import pickle
import random
import gc

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding,
                          set_seed as hf_set_seed)
from noise import corrupt_text, ERROR_TYPES

SEED = 42
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
NAMES = {'word_tfidf': 'Word TF-IDF + LR', 'char_tfidf': 'Char TF-IDF + LR',
         'distilbert': 'DistilBERT', 'distilbert_augmented': 'DistilBERT + typo augmentation'}
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    hf_set_seed(SEED)


def load_data():
    data = load_dataset('ag_news')
    assert len(data['train']) == 120000 and len(data['test']) == 7600
    split = data['train'].train_test_split(test_size=10000, seed=SEED,
                                          stratify_by_column='label')
    train, validation, test = split['train'], split['test'], data['test']
    for part in (train, validation, test):
        assert set(part['label']) == {0, 1, 2, 3}
    assert len(train) == 110000 and len(validation) == 10000
    sample = 'Market prices climbed, today!'
    assert corrupt_text(sample, 0, random.Random(SEED)) == sample
    assert (corrupt_text(sample, 1, random.Random(SEED)) ==
            corrupt_text(sample, 1, random.Random(SEED)))
    print('Data and noise sanity checks passed.', flush=True)
    return train, validation, test


def build_noisy_test_sets(texts):
    variants = {}
    for p in LEVELS:
        rng = random.Random(SEED)
        variants[p] = [corrupt_text(text, p, rng) for text in texts]
    for kind in ERROR_TYPES:
        rng = random.Random(SEED)
        variants[kind] = [corrupt_text(text, 0.20, rng, kind) for text in texts]
    assert variants[0.0] == texts
    return variants


def train_tfidf(train, character=False):
    name = 'char_tfidf' if character else 'word_tfidf'
    path = CHECKPOINTS / f'{name}.pkl'
    if path.exists():
        with path.open('rb') as file:
            return pickle.load(file)
    vectorizer = TfidfVectorizer(
        analyzer='char' if character else 'word',
        ngram_range=(3, 5) if character else (1, 2),
        max_features=150000 if character else 100000, min_df=2, sublinear_tf=True)
    model = make_pipeline(vectorizer, LogisticRegression(
        C=4.0, max_iter=1000, solver='saga', n_jobs=-1, random_state=SEED))
    print(f'Training {name}', flush=True)
    model.fit(list(train['text']), train['label'])
    with path.open('wb') as file:
        pickle.dump(model, file)
    return model


def tokenize(texts, labels, tokenizer):
    data = Dataset.from_dict({'text': list(texts), 'labels': list(labels)})
    return data.map(lambda batch: tokenizer(batch['text'], truncation=True,
                                            max_length=128),
                    batched=True, remove_columns=['text'])


def train_distilbert(train, validation, augmented=False):
    set_seed()
    name = 'distilbert_augmented' if augmented else 'distilbert'
    saved = CHECKPOINTS / name
    source = str(saved) if (saved / 'complete').exists() else 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=4)
    args = TrainingArguments(
        output_dir=str(CHECKPOINTS / f'{name}_training'), num_train_epochs=3,
        learning_rate=2e-5, weight_decay=0.01, per_device_train_batch_size=32,
        per_device_eval_batch_size=64, seed=SEED, data_seed=SEED,
        eval_strategy='epoch', save_strategy='no', report_to='none',
        logging_steps=200, dataloader_pin_memory=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args,
                      eval_dataset=tokenize(validation['text'], validation['label'], tokenizer),
                      data_collator=DataCollatorWithPadding(tokenizer),
                      processing_class=tokenizer)
    if not (saved / 'complete').exists():
        texts = list(train['text'])
        if augmented:
            rng = random.Random(SEED)
            texts = [corrupt_text(text, rng.uniform(0.05, 0.20), rng)
                     if rng.random() < 0.5 else text for text in texts]
        trainer.train_dataset = tokenize(texts, train['label'], tokenizer)
        print(f'Training {name} on {trainer.args.device}', flush=True)
        trainer.train()
        trainer.save_model(str(saved))
        tokenizer.save_pretrained(str(saved))
        (saved / 'complete').write_text('Finished three epochs, seed 42.\n')
    else:
        print(f'Loaded {name} on {trainer.args.device}', flush=True)

    def predict(texts):
        data = tokenize(texts, [0] * len(texts), tokenizer).remove_columns('labels')
        return trainer.predict(data).predictions.argmax(axis=-1)
    return predict


def evaluate_model(name, predict, variants, labels):
    rows, errors = [], []
    for variant, texts in variants.items():
        predictions = np.asarray(predict(texts))
        assert predictions.shape == (len(labels),)
        assert np.isin(predictions, [0, 1, 2, 3]).all()
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro', labels=[0, 1, 2, 3], zero_division=0)
        assert np.isfinite([accuracy, f1]).all()
        if variant == 0.0 and (len(np.unique(predictions)) == 1 or accuracy <= 0.30):
            raise RuntimeError(f'{name}: investigate technically broken clean predictions')
        row = dict(model=name, accuracy=accuracy, macro_f1=f1)
        if isinstance(variant, str):
            errors.append(dict(row, error_type=variant))
        else:
            rows.append(dict(row, noise_probability=variant))
        print(f'{name} {variant}: accuracy={accuracy:.6f}, macro_f1={f1:.6f}', flush=True)
    return rows, errors


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|---|' + '---:|' * (len(headers) - 1)] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def save_results(rows, errors):
    main = pd.DataFrame(rows)[['model', 'noise_probability', 'accuracy', 'macro_f1']]
    error = pd.DataFrame(errors)[['model', 'error_type', 'accuracy', 'macro_f1']]
    main.to_csv(RESULTS / 'main_results.csv', index=False)
    error.to_csv(RESULTS / 'error_types.csv', index=False)
    summary_rows = []
    for name, group in main.groupby('model', sort=False):
        values = group.set_index('noise_probability').macro_f1
        summary_rows.append(dict(model=name, clean_macro_f1=values[0.0],
                                 macro_f1_at_30=values[0.30],
                                 retention_at_30=values[0.30] / values[0.0],
                                 mean_macro_f1=values.mean()))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / 'summary.csv', index=False)
    # Keep measured partial CSVs if a later training run is interrupted.
    if len(summary) != 4:
        return
    table = main.pivot(index='model', columns='noise_probability', values='macro_f1')
    error_table = error.pivot(index='model', columns='error_type', values='macro_f1')
    main_md = markdown_table(['Model', '0%', '5%', '10%', '20%', '30%'],
                            [[NAMES[n]] + [f'{table.loc[n, p]:.4f}' for p in LEVELS] for n in NAMES])
    summary_md = markdown_table(['Model', 'Clean F1', 'F1 at 30%', 'Retention'],
        [[NAMES[r.model], f'{r.clean_macro_f1:.4f}', f'{r.macro_f1_at_30:.4f}',
          f'{r.retention_at_30:.4f}'] for r in summary.itertuples()])
    error_md = markdown_table(['Model', 'Delete', 'Swap', 'Keyboard', 'Duplicate'],
        [[NAMES[n]] + [f'{error_table.loc[n, k]:.4f}' for k in ERROR_TYPES] for n in NAMES])
    best_clean = table[0.0].max()
    winners = ', '.join(NAMES[n] for n in table.index[table[0.0] == best_clean])
    best_noisy = table[0.30].max()
    noisy_winners = ', '.join(NAMES[n] for n in table.index[table[0.30] == best_noisy])
    observations = [f'- Highest clean Macro-F1: {winners} ({best_clean:.4f}).',
                    f'- Highest Macro-F1 at p=0.30: {noisy_winners} ({best_noisy:.4f}).']
    for p in (0.0, 0.30):
        delta = table.loc['distilbert_augmented', p] - table.loc['distilbert', p]
        observations.append(f'- Augmentation changed Macro-F1 at p={p:.2f} by {delta:+.4f} (augmented minus baseline).')
    (RESULTS / 'report_values.md').write_text(
        '# Results for report\n\nAll scores use the 0–1 scale. Main and error-type tables show Macro-F1.\n\n'
        '## Main results\n\n' + main_md + '\n\n## Robustness summary\n\n' + summary_md +
        '\n\n## Error types at p=0.20\n\n' + error_md +
        '\n\n## Short observations\n\n' + '\n'.join(observations) + '\n')
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in NAMES:
        ax.plot(LEVELS, table.loc[name, LEVELS], marker='o', label=NAMES[name])
    ax.set(xlabel='Noise probability', ylabel='Macro-F1', xticks=LEVELS)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / 'robustness_curve.png', dpi=160)
    plt.close(fig)
    readme = ROOT / 'README.md'
    content = readme.read_text()
    start, rest = content.split('<!-- RESULTS START -->', 1)
    _, end = rest.split('<!-- RESULTS END -->', 1)
    readme.write_text(start + '<!-- RESULTS START -->\nMacro-F1 (0–1 scale):\n\n' + main_md +
                      '\n<!-- RESULTS END -->' + end.replace(
                          'The run generates `results/robustness_curve.png`.',
                          '![Robustness curve](results/robustness_curve.png)'))
    print('\nMain results\n', main.to_string(index=False))
    print('\nRobustness summary\n', summary.to_string(index=False))
    print('\nError-type results\n', error.to_string(index=False))
    print('\nResults saved to results/', flush=True)


def main():
    set_seed()
    RESULTS.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}; PyTorch {torch.__version__}', flush=True)
    train, validation, test = load_data()
    variants = build_noisy_test_sets(list(test['text']))
    labels = np.array(test['label'])
    rows, errors = [], []
    for name in NAMES:
        if name.endswith('tfidf'):
            model = train_tfidf(train, character=name == 'char_tfidf')
            predict = model.predict
        else:
            predict = train_distilbert(train, validation, augmented=name.endswith('augmented'))
        new_rows, new_errors = evaluate_model(name, predict, variants, labels)
        rows.extend(new_rows)
        errors.extend(new_errors)
        save_results(rows, errors)
        del predict
        if name.endswith('tfidf'):
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
