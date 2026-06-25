# Bao cao ngan ve huan luyen mo hinh

## 1. Bo sieu tham so da chon

Sau khi thu nghiem o Buoc 1, mo hinh ban dau voi `RandomForestClassifier` va bo tham so:

```yaml
n_estimators: 200
max_depth: 10
min_samples_split: 5
```

cho ket qua `accuracy = 0.6440`, thap hon nguong yeu cau `0.70`. Vi vay, em chon cau hinh moi nhu sau:

```yaml
model_type: extra_trees
extra_data_paths:
  - data/train_phase2.csv
n_estimators: 700
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
random_state: 42
n_jobs: -1
```

Ly do lua chon:

- Su dung them `train_phase2.csv` giup tang kich thuoc tap huan luyen tu 2.998 mau len 5.996 mau, giup mo hinh hoc duoc nhieu mau du lieu hon.
- `ExtraTreesClassifier` phu hop voi bai toan du lieu dang bang va cho ket qua tot hon Random Forest trong qua trinh thu nghiem.
- Tang `n_estimators` len `700` giup mo hinh on dinh hon do ket hop nhieu cay quyet dinh.
- Dat `max_depth: null` cho phep cay hoc linh hoat hon, phu hop khi du lieu da duoc tang them.
- Dat `max_features: sqrt` giup moi cay chi xem xet mot phan dac trung, lam tang tinh da dang giua cac cay va giam overfitting.
- Dat `random_state: 42` de ket qua co the lap lai, va `n_jobs: -1` de tan dung tat ca CPU khi huan luyen.

Voi cau hinh moi, ket qua dat duoc:

```text
Accuracy: 0.7600
F1-score: 0.7590
Passed threshold: true
```

Ket qua nay vuot nguong danh gia `0.70`, nen mo hinh du dieu kien qua eval gate.

## 2. Kho khan va cach giai quyet

Kho khan dau tien la accuracy ban dau chi dat `0.6440`, khong du nguong deploy. Em da thu nghiem cac cach tang do chinh xac nhu tang do sau cua Random Forest, tang so cay, can bang lop va doi sang mo hinh ensemble khac. Ket qua cho thay neu chi dung `train_phase1.csv` thi accuracy van chua vuot `0.70`; khi bo sung `train_phase2.csv` va dung `ExtraTreesClassifier`, accuracy tang len `0.7600`.

Kho khan thu hai la MLflow bi loi do thu muc `mlruns/0` thieu file `meta.yaml`, lam qua trinh train va unit test bi dung truoc khi tinh metric. Em da cau hinh lai tracking URI sang `sqlite:///mlflow.db` va dat experiment mac dinh trong ham `train()`, giup MLflow khoi tao on dinh hon trong ca local va CI.

Kho khan cuoi cung la CI ban dau chi `dvc pull` hai file `train_phase1.csv` va `eval.csv`, trong khi cau hinh moi can them `train_phase2.csv`. Em da cap nhat workflow de pull ca ba file du lieu, dam bao pipeline tren GitHub Actions co du du lieu de huan luyen lai mo hinh.
