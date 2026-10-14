# Datasets

AutoPolish is evaluated on four public datasets. None of them is redistributed in this
repository: each is published by its authors under its own terms, so download them from the
official source and place them under `data/` as shown below. Everything under `data/` is
git-ignored.

You only need the source datasets for **tier 3** of the [reproduction guide](REPRODUCING.md)
(re-running the models). Tiers 1 and 2 work from the [experiment outputs](#experiment-outputs)
alone.

| Dataset | What it contains | Used for | Terms | Goes to |
|---|---|---|---|---|
| [PARA](https://web.xidian.edu.cn/ldli/en/dataset.html) | photographs, 1–5 ratings, rater demographics and Big-Five personality | persona judge, editing loop | access on request from the authors | `data/para/` |
| [EVA](https://github.com/kang-gnak/eva-dataset) | AVA photographs, 0–10 ratings, rater age, gender, region and photographic skill | persona judge, editing loop | CC0-1.0 | `data/eva/` |
| [LAPIS](https://sites.google.com/view/lapisdataset/homepage) | paintings, 0–100 ratings, rater nationality, art interest and age | persona judge | terms of use form | `data/lapis/` |
| [Rapidata](https://huggingface.co/datasets/Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3) | pairs of AI-generated images, pairwise votes with voter country and language | generated-image transfer | CDLA-Permissive-2.0 | `data/rapidata_700k/` |

## Expected layout

```
data/
├── para/
│   ├── annotation/          PARA-Images.csv, PARA-UserInfo.csv
│   └── imgs/<session>/<image>.jpg
├── eva/
│   ├── data/                votes_filtered.csv, users.csv, region_index.csv  (`=`-separated)
│   └── images/<image_id>.jpg
├── lapis/
│   ├── annotation/          LAPIS_PIAA.csv
│   └── images/<image_filename>
└── rapidata_700k/
    └── data/train_*.parquet
```

## PARA

Request access on the [PARA dataset page](https://web.xidian.edu.cn/ldli/en/dataset.html). The
authors provide a download link and an archive password once you agree to their terms. Extract
the archive so that the annotation tables sit in `data/para/annotation/` and the images keep
their per-session folders under `data/para/imgs/`.

## EVA

EVA is stored with Git LFS, so install [git-lfs](https://git-lfs.com) first.

```bash
git clone https://github.com/kang-gnak/eva-dataset /tmp/eva-dataset
mkdir -p data/eva/images
cp -r /tmp/eva-dataset/data data/eva/
cd /tmp/eva-dataset/images && cat EVA_together.zip.00* > EVA_together.zip && unzip -q EVA_together.zip
mv /tmp/eva-dataset/images/EVA_together/*.jpg "$OLDPWD/data/eva/images/"
```

The archive holds 5,101 resized AVA photographs, of which 4,070 appear in `votes_filtered.csv`.
EVA images are a subset of AVA (file names are AVA ids), which matters if you combine the two.

## LAPIS

Fill in the terms of use form on the [LAPIS homepage](https://sites.google.com/view/lapisdataset/homepage)
to receive the archive password, then download `LAPIS password protected.zip` from
[OSF](https://osf.io/zw39r/). Move its `annotation/` and `images/` folders to `data/lapis/`.

> **macOS note.** Nine image file names in the archive use a double-encoded character set, and
> `unzip` stops on them with "Illegal byte sequence". Extract with Python instead, which yields
> valid UTF-8 names:
>
> ```bash
> python -c "import zipfile; zipfile.ZipFile('LAPIS password protected.zip').extractall('lapis_raw', pwd=b'<password>')"
> ```
>
> A handful of names still cannot match the CSV byte for byte on APFS. The pipeline drops those
> images at load time with a warning, which is expected.

## Rapidata

```bash
huggingface-cli download Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3 \
    --repo-type dataset --local-dir data/rapidata_700k
```

The pipeline reads the parquet shards directly, so there is no preparation step. Please cite
the dataset as its card requests: Christodoulou and Kuhlmann-Jørgensen,
[arXiv:2409.11904](https://arxiv.org/abs/2409.11904).

## Experiment outputs

Our own outputs are on the Hugging Face Hub at
[`savoji/AUTOPOLISH`](https://huggingface.co/datasets/savoji/AUTOPOLISH): every analysis JSON and
figure, the raw logs of every judge run and the edited images of every editing run (~2.2 GB).

```bash
python tools/fetch_from_hf.py autopolish
```

This restores two trees, `results/` (analysis outputs) and `data/results/` (raw run logs and
edited images). The download lands in the Hugging Face cache before it is copied into place, so
budget about twice its size in free disk space, and run `huggingface-cli delete-cache`
afterwards to reclaim the cache copy.

The edited images are derivatives of PARA and EVA photographs, and the run logs contain persona
descriptions built from the PARA, EVA and LAPIS rater attributes. Use them under the terms of
the source datasets above.
