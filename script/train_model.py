import argparse
import functools
import gc
import json
import os
import pickle
import time

import git
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit
import sys
sys.path.append(r'D:\04_project\03_RNA2ADT\open-problems-multimodal')

from ss_opm.metric.correlation_score import correlation_score
from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder
from ss_opm.pre_post_processing.pre_post_processing import PrePostProcessing
from ss_opm.utility.get_group_id import get_group_id
from ss_opm.utility.get_metadata_pattern import get_metadata_pattern
from ss_opm.utility.get_selector_with_metadata_pattern import get_selector_with_metadata_pattern
from ss_opm.utility.load_dataset import load_dataset
from ss_opm.utility.row_normalize import row_normalize
from ss_opm.utility.set_seed import set_seed


def get_params_default(trial):
    params = {}
    return params


def _build_model_default(params):
    model = Ridge(**params)
    return model


def _build_pre_post_process_default(params):
    pre_post_process = PrePostProcessing(params)
    return pre_post_process


class CrossVaridation(object):
    def __init__(self):
        pass

    def compute_score(
        self,
        x,
        y,
        metadata,
        x_test,
        metadata_test,
        params,
        build_model=_build_model_default,
        build_pre_post_process=_build_pre_post_process_default,
        dump=False,
        dump_dir="./",
        n_splits=3,
        n_bagging=0,
        bagging_ratio=1.0,
        use_batch_group=True,
    ):
        groups = None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        if use_batch_group:
            batch_groups = get_group_id(metadata)
            n_groups = len(np.unique(batch_groups))
            if n_groups > n_splits:
                print(f"use batch group. # groups is {n_groups}")
                groups = batch_groups
                kf = GroupKFold(n_splits=n_splits)
        pre_post_processes = []
        models = []
        cell_types = metadata["cell_type"].unique()
        scores_dict = {
            "mse_train": [],
            "corrscore_train": [],
            "mse_val": [],
            "corrscore_val": [],
        }

        for cell_type_name in cell_types:
            scores_dict[f"{cell_type_name}_mse_train"] = []
            scores_dict[f"{cell_type_name}_corrscore_train"] = []
            scores_dict[f"{cell_type_name}_mse_val"] = []
            scores_dict[f"{cell_type_name}_corrscore_val"] = []
        print("kfold type", type(kf), "group", groups)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(x, y, groups=groups)):
            if groups is not None:
                print("train group:", np.unique(groups[idx_tr]))
                print("val group:", np.unique(groups[idx_va]))
            start_time = time.time()
            model = None
            score = {}
            gc.collect()
            _n_bagging = n_bagging
            if _n_bagging == 0:
                _n_bagging = 1
            y_train_pred = None
            y_val_pred = None
            metadata_train = metadata.iloc[idx_tr, :]
            if "selected_metadata" in params["model"]:
                selector = get_selector_with_metadata_pattern(
                    metadata=metadata_train, metadata_pattern=params["model"]["selected_metadata"]
                )
                if np.sum(selector) == 0:
                    print("skip!")
                    continue
            x_train = x[idx_tr]  # creates a copy, https://numpy.org/doc/stable/user/basics.copies.html
            y_train = y[idx_tr].toarray()
            # We validate the model
            x_val = x[idx_va]
            y_val = y[idx_va].toarray()
            metadata_val = metadata.iloc[idx_va, :]
            for bagging_i in range(_n_bagging):
                gc.collect()
                if n_bagging > 0:
                    print("bagging_i", bagging_i, flush=True)
                    n_bagging_size = int(x_train.shape[0] * bagging_ratio)
                    bagging_idx = np.random.permutation(x_train.shape[0])[:n_bagging_size]
                    x_train_bagging = x_train[bagging_idx]
                    y_train_bagging = y_train[bagging_idx]
                    metadata_train_bagging = metadata_train.iloc[bagging_idx, :]
                else:
                    x_train_bagging = x_train
                    y_train_bagging = y_train
                    metadata_train_bagging = metadata_train

                pre_post_process = build_pre_post_process(params=params["pre_post_process"])
                if not pre_post_process.is_fitting:
                    pre_post_process.fit_preprocess(
                        inputs_values=x_train_bagging,
                        targets_values=y_train_bagging,
                        metadata=metadata_train_bagging,
                    )
                else:
                    print("skip pre_post_process fit")
                pre_post_processes.append(pre_post_process)
                preprocessed_x_train, preprocessed_y_train = pre_post_process.preprocess(
                    inputs_values=x_train_bagging,
                    targets_values=y_train_bagging,
                    metadata=metadata_train_bagging,
                )
                model = build_model(params=params["model"])
                print(f"model input shape X:{preprocessed_x_train.shape} Y:{preprocessed_y_train.shape}")
                model.fit(
                    x=x_train_bagging,
                    y=y_train_bagging,
                    preprocessed_x=preprocessed_x_train,
                    preprocessed_y=preprocessed_y_train,
                    metadata=metadata_train_bagging,
                    pre_post_process=pre_post_process,
                )
                preprocessed_y_train_pred = model.predict(x=x_train, preprocessed_x=preprocessed_x_train, metadata=metadata_train)
                new_y_train_pred = pre_post_process.postprocess(preprocessed_y_train_pred)
                preprocessed_x_val, _ = pre_post_process.preprocess(
                    inputs_values=x_val,
                    targets_values=None,
                    metadata=metadata_val,
                )
                preprocessed_y_val_pred = model.predict(x=x_val, preprocessed_x=preprocessed_x_val, metadata=metadata_val)
                new_y_val_pred = pre_post_process.postprocess(preprocessed_y_val_pred)
                models.append(model)

                mse_train = mean_squared_error(y_train, new_y_train_pred)
                corrscore_train = correlation_score(y_train, new_y_train_pred)
                mse_val = mean_squared_error(y_val, new_y_val_pred)
                corrscore_val = correlation_score(y_val, new_y_val_pred)
                print(
                    f"Fold {fold} bagging {bagging_i} "
                    f"mse_train: {mse_train: .5f} "
                    f"corrscore_train: {corrscore_train: .5f} "
                    f"mse_val: {mse_val: .5f} "
                    f"corrscore_val: {corrscore_val: .5f} "
                )

                if y_train_pred is None:
                    y_train_pred = new_y_train_pred
                else:
                    y_train_pred += new_y_train_pred
                if y_val_pred is None:
                    y_val_pred = new_y_val_pred
                else:
                    y_val_pred += new_y_val_pred

            y_train_pred /= _n_bagging
            y_val_pred /= _n_bagging

            mse_train = mean_squared_error(y_train, y_train_pred)
            corrscore_train = correlation_score(y_train, y_train_pred)
            score["mse_train"] = mse_train
            score["corrscore_train"] = corrscore_train
            cell_type_train = metadata_train["cell_type"].values
            for cell_type_name in cell_types:
                s = cell_type_train == cell_type_name
                if s.sum() > 10:
                    mse_train = mean_squared_error(y_train[s], y_train_pred[s])
                    corrscore_train = correlation_score(y_train[s], y_train_pred[s])
                    score[f"{cell_type_name}_mse_train"] = mse_train
                    score[f"{cell_type_name}_corrscore_train"] = corrscore_train
                else:
                    score[f"{cell_type_name}_mse_train"] = 0.0
                    score[f"{cell_type_name}_corrscore_train"] = 1.0
            gc.collect()

            mse_val = mean_squared_error(y_val, y_val_pred)
            corrscore_val = correlation_score(y_val, y_val_pred)
            score["mse_val"] = mse_val
            score["corrscore_val"] = corrscore_val
            cell_type_val = metadata_val["cell_type"].values
            for cell_type_name in cell_types:
                s = cell_type_val == cell_type_name
                if s.sum() > 10:
                    mse_val = mean_squared_error(y_val[s], y_val_pred[s])
                    corrscore_val = correlation_score(y_val[s], y_val_pred[s])
                    score[f"{cell_type_name}_mse_val"] = mse_val
                    score[f"{cell_type_name}_corrscore_val"] = corrscore_val
                else:
                    score[f"{cell_type_name}_mse_val"] = 0.0
                    score[f"{cell_type_name}_corrscore_val"] = 1.0
            if dump:
                np.save(os.path.join(dump_dir, f"k{fold}_y_val.npy"), y_val)
                np.save(os.path.join(dump_dir, f"k{fold}_y_val_pred.npy"), y_val_pred)

            del x_train, y_train, y_train_pred, x_val, y_val_pred, y_val
            print(f"Fold {fold}: score:{score} elapsed time = {time.time() - start_time: .3f}")
            for k, v in score.items():
                scores_dict[k].append(v)

        # Show overall score
        result_df = pd.DataFrame(scores_dict)
        return result_df, models, pre_post_processes


class Objective(object):
    def __init__(
        self,
        x,
        y,
        metadata,
        x_test,
        metadata_test,
        test_ratio=0.2,
        get_params=get_params_default,
        build_model=_build_model_default,
        build_pre_post_process=_build_pre_post_process_default,
    ):
        self.test_ratio = test_ratio
        splitter = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        train_index, val_index = next(splitter.split(x))

        self.x_train = x[train_index, :]
        self.x_val = x[val_index, :]
        self.y_train = y[train_index, :]
        self.y_val = y[val_index, :].toarray()
        self.metadata_train = metadata.iloc[train_index, :]
        self.metadata_val = metadata.iloc[val_index, :]
        self.get_params = get_params
        self.build_model = build_model
        self.build_pre_post_process = build_pre_post_process

    def __call__(self, trial):
        gc.collect()
        params = self.get_params(trial=trial)
        pre_post_process = self.build_pre_post_process(params=params["pre_post_process"])
        model = self.build_model(params=params["model"])
        if not pre_post_process.is_fitting:
            pre_post_process.fit_preprocess(inputs_values=self.x_train, targets_values=self.y_train)
        else:
            print("skip pre_post_process fit")
        preprocessed_inputs_values, preprocessed_targets_values = pre_post_process.preprocess(
            inputs_values=self.x_train,
            targets_values=self.y_train,
            metadata=self.metadata_train,
        )

        print(f"model input shape X:{preprocessed_inputs_values.shape} Y:{preprocessed_targets_values.shape}")
        model.fit(
            preprocessed_x=preprocessed_inputs_values,
            preprocessed_y=preprocessed_targets_values,
            x=self.x_train,
            y=self.y_train,
            metadata=self.metadata_train,
            pre_post_process=pre_post_process,
        )
        preprocessed_inputs_values, _ = pre_post_process.preprocess(
            inputs_values=self.x_val,
            targets_values=None,
            metadata=self.metadata_val,
        )
        preprocessed_y_pred_val = model.predict(
            preprocessed_x=preprocessed_inputs_values, x=self.x_val, metadata=self.metadata_val
        )
        y_val_pred = pre_post_process.postprocess(preprocessed_y_pred_val)
        # print("y_test_pred", y_test_pred)
        corrscore = correlation_score(self.y_val, y_val_pred)

        return corrscore


# def main():
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=r'D:\04_project\03_RNA2ADT\data\processed')
parser.add_argument("--task_type", default="cite", choices=["multi", "cite"])
parser.add_argument("--cell_type", default="all", choices=["all", "hsc", "eryp", "neup", "masp", "mkp", "bp", "mop"])
parser.add_argument("--n_trials", type=int, default=0)
parser.add_argument("--skip_test_prediction", action="store_true")
parser.add_argument("--n_model_train_samples", type=int, default=-1)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--distributed_study_name")
parser.add_argument("--model", default="ead")
parser.add_argument("--snapshot", default=None)
parser.add_argument("--param_path", metavar="PATH")
parser.add_argument("--out_dir", metavar="PATH", default=r"D:\04_project\03_RNA2ADT\open-problems-multimodal\result")
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--cv_dump", action="store_true")
parser.add_argument("--cv_n_bagging", type=int, default=0)
parser.add_argument("--use_k_fold_models", action="store_true")
parser.add_argument("--n_splits", type=int, default=3)
parser.add_argument("--pre_post_process_tuning", action="store_true")
parser.add_argument("--check_load_model", action="store_true")
parser.add_argument("--metadata_pattern_id", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

try:
    git_hexsha = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True).head.commit.hexsha
except Exception:
    git_hexsha = "Failed to get git"
print("git_hexsha", git_hexsha)
if args.metadata_pattern_id is not None:
    print("metadata_pattern_id", args.metadata_pattern_id)
    print(get_metadata_pattern(metadata_pattern_id=args.metadata_pattern_id))

set_seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)
cell_type_names = {
    "all": "all",
    "hsc": "HSC",
    "eryp": "EryP",
    "neup": "NeuP",
    "masp": "MasP",
    "mkp": "MkP",
    "bp": "BP",
    "mop": "MoP",
}
data_dir = args.data_dir

train_inputs, train_metadata, train_target = load_dataset(
    data_dir=data_dir, task_type=args.task_type, split="train", cell_type=cell_type_names[args.cell_type]
)
test_inputs, test_metadata, _ = load_dataset(data_dir=data_dir, task_type=args.task_type, split="test")

if args.model == "ead":
    model_class = EncoderDecoder
    pre_post_process_class = PrePostProcessing
else:
    raise ValueError

loaded_params = None
if args.param_path is not None:
    with open(os.path.join(args.param_path)) as f:
        loaded_params = json.load(f)

def get_params_core(trial, pre_post_process_tuning=False):
    if loaded_params is not None:
        return loaded_params
    model_params = model_class.get_params(
        task_type=args.task_type,
        device=args.device,
        trial=trial,
        debug=args.debug,
        snapshot=args.snapshot,
        metadata_pattern_id=args.metadata_pattern_id,
    )
    if pre_post_process_tuning:
        pre_post_process_params = pre_post_process_class.get_params(
            task_type=args.task_type,
            trial=trial,
            debug=args.debug,
            seed=args.seed,
        )
    else:
        pre_post_process_params = pre_post_process_class.get_params(
            task_type=args.task_type,
            data_dir=args.data_dir,
            trial=None,
            debug=args.debug,
            seed=args.seed,
        )
    return {"model": model_params, "pre_post_process": pre_post_process_params}

get_params = functools.partial(get_params_core, pre_post_process_tuning=args.pre_post_process_tuning)

params = get_params(trial=None)
pre_post_process_default = None
# if not args.pre_post_process_tuning:
#     pre_post_process_default = pre_post_process_class(params["pre_post_process"])
#     pre_post_process_default.fit_preprocess(
#         inputs_values=train_inputs,
#         targets_values=train_target,
#         metadata=train_metadata,
#         test_inputs_values=test_inputs,
#         test_metadata=test_metadata,
#     )

from pathlib import Path
import hashlib, json, pickle, os

def _hashable(o):
    try:
        return json.dumps(o, sort_keys=True, default=str)
    except Exception:
        return str(o)

def _make_ppp_cache_key(args, params, train_inputs, train_target):
    key = {
        "task_type": args.task_type,
        "cell_type": getattr(args, "cell_type", None),
        "data_dir": os.path.abspath(args.data_dir),
        "use_test_inputs": params["pre_post_process"]["use_test_inputs"],
        "inputs_decomposer_method": params["pre_post_process"]["inputs_decomposer_method"],
        "inputs_decomposer": params["pre_post_process"]["inputs_decomposer"],
        "targets_decomposer_method": params["pre_post_process"]["targets_decomposer_method"],
        "targets_decomposer": params["pre_post_process"]["targets_decomposer"],
        "use_targets_decomposer": params["pre_post_process"]["use_targets_decomposer"],
        "use_targets_normalization": params["pre_post_process"]["use_targets_normalization"],
        "use_inputs_scaler": params["pre_post_process"]["use_inputs_scaler"],
        "use_targets_scaler": params["pre_post_process"]["use_targets_scaler"],
        # 形状也纳入，确保数据维度一致
        "x_shape": tuple(train_inputs.shape),
        "y_shape": tuple(train_target.shape),
        # 可选：随机种子
        "seed": params.get("seed", 42),
    }
    s = _hashable(key)
    return hashlib.md5(s.encode()).hexdigest()

cache_dir = Path(getattr(args, "out_dir", ".")) / "cache_ppp"
cache_dir.mkdir(parents=True, exist_ok=True)

ppp_hash = _make_ppp_cache_key(args, params, train_inputs, train_target)
ppp_path = cache_dir / f"pre_post_process_{ppp_hash}.pkl"

ppp_path = Path(r'D:\04_project\03_RNA2ADT\open-problems-multimodal\result\cache_ppp\pre_post_process_8af95395adcb91c973757800fc749ee4.pkl')
pre_post_process_default = None
if not args.pre_post_process_tuning:
    if ppp_path.exists():
        print(f"[PPP] Loading cached PrePostProcessing from {ppp_path}")
        with open(ppp_path, "rb") as f:
            pre_post_process_default = pickle.load(f)
        pre_post_process_default.is_fitting = True  # 标记成已拟合
    else:
        print("[PPP] Fitting PrePostProcessing (no cache). This may take a while...")
        pre_post_process_default = pre_post_process_class(params["pre_post_process"])
        pre_post_process_default.fit_preprocess(
            inputs_values=train_inputs,
            targets_values=train_target,
            metadata=train_metadata,
            test_inputs_values=test_inputs,
            test_metadata=test_metadata,
        )
        with open(ppp_path, "wb") as f:
            pickle.dump(pre_post_process_default, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[PPP] Cached to {ppp_path}")

def build_pre_post_process(params):
    if pre_post_process_default is not None:
        return pre_post_process_default
    else:
        pre_post_process = pre_post_process_class(params)
        return pre_post_process

def build_model(params):
    model = model_class(params)
    return model

if args.n_model_train_samples > 0:
    np.random.seed(42)
    all_row_indices = np.arange(train_inputs.shape[0])
    np.random.shuffle(all_row_indices)
    n_model_train_samples = args.n_model_train_samples
    if n_model_train_samples > train_inputs.shape[0]:
        n_model_train_samples = train_inputs.shape[0]
    selected_rows_indices = all_row_indices[:n_model_train_samples]
    train_inputs = train_inputs[selected_rows_indices]
    train_metadata = train_metadata.iloc[selected_rows_indices]
    train_target = train_target[selected_rows_indices]
print("train sample size:", train_inputs.shape[0])

if args.n_trials > 0:
    study_name = None
    storage = "sqlite:///{}/optuna.db".format(args.out_dir)
    if args.distributed_study_name is not None:
        study_name = args.distributed_study_name
        storage = optuna.storages.RDBStorage(os.environ["OPTUNA_STORAGE"], {"pool_pre_ping": True})
    print("study_name", study_name)
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, constant_liar=True),
        direction="maximize",
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    objective = Objective(
        x=train_inputs,
        y=train_target,
        metadata=train_metadata,
        x_test=test_inputs,
        metadata_test=test_metadata,
        test_ratio=0.2,
        get_params=get_params,
        build_model=build_model,
        build_pre_post_process=build_pre_post_process,
    )
    while len(study.get_trials()) < args.n_trials:
        study.optimize(objective, n_trials=1)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    best_params = get_params(study.best_trial)
    print(json.dumps(best_params, indent=2))
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))
    params = best_params
    del objective

print("dump params")
with open(os.path.join(args.out_dir, "params.json"), "w") as f:
    json.dump(params, f, indent=2)

cv = CrossVaridation()
result_df, k_fold_models, k_fold_pre_post_processes = cv.compute_score(
    x=train_inputs,
    y=train_target,
    metadata=train_metadata,
    x_test=test_inputs,
    metadata_test=test_metadata,
    build_model=build_model,
    build_pre_post_process=build_pre_post_process,
    params=params,
    n_splits=args.n_splits,
    dump=args.cv_dump,
    dump_dir=args.out_dir,
    n_bagging=args.cv_n_bagging,
)
print("Average:", result_df.mean(), flush=True)
del cv
gc.collect()


if not args.skip_test_prediction:
    if args.use_k_fold_models:
        print("use k fold models for test preds")
        start_time = time.time()
        y_test_pred = None
        print("pridict with test data", flush=True)
        for _, (model, pre_post_process) in enumerate(zip(k_fold_models, k_fold_pre_post_processes)):
            preprocessed_test_inputs, _ = pre_post_process.preprocess(
                inputs_values=test_inputs, targets_values=None, metadata=test_metadata
            )
            preprocessed_y_test_pred = model.predict(
                x=test_inputs, preprocessed_x=preprocessed_test_inputs, metadata=test_metadata
            )
            new_y_test_pred = pre_post_process.postprocess(preprocessed_y_test_pred)
            if y_test_pred is None:
                y_test_pred = row_normalize(new_y_test_pred)
            else:
                y_test_pred += row_normalize(new_y_test_pred)
            y_test_pred /= len(k_fold_models)
        print(f"elapsed time = {time.time() - start_time: .3f}")
    else:
        print("train model to predict with test data", flush=True)
        start_time = time.time()
        pre_post_process = build_pre_post_process(params["pre_post_process"])
        if not pre_post_process.is_fitting:
            pre_post_process.fit_preprocess(inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata)
        else:
            print("skip pre_post_process fit")
        preprocessed_inputs_values, preprocessed_targets_values = pre_post_process.preprocess(
            inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata
        )
        preprocessed_test_inputs, _ = pre_post_process.preprocess(
            inputs_values=test_inputs, targets_values=None, metadata=test_metadata
        )
        model = build_model(params=params["model"])
        model.fit(
            x=train_inputs,
            y=train_target,
            preprocessed_x=preprocessed_inputs_values,
            preprocessed_y=preprocessed_targets_values,
            metadata=train_metadata,
            pre_post_process=pre_post_process,
        )
        print(f"elapsed time = {time.time() - start_time: .3f}")
        print("pridict with test data", flush=True)
        start_time = time.time()
        preprocessed_y_test_pred = model.predict(
            x=test_inputs, preprocessed_x=preprocessed_test_inputs, metadata=test_metadata
        )
        y_test_pred = pre_post_process.postprocess(preprocessed_y_test_pred)
        print(f"elapsed time = {time.time() - start_time: .3f}")
        print("dump preprocess and model")
        model_dir = os.path.join(args.out_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "pre_post_process.pickle"), "wb") as f:
            pickle.dump(pre_post_process, f)
        model.save(model_dir)
        if args.check_load_model:
            with open(os.path.join(model_dir, "pre_post_process.pickle"), "rb") as f:
                _ = pickle.load(f)
            loaded_model = build_model(params=None)
            loaded_model.load(model_dir)
    if args.cell_type != "all":
        s = test_metadata["cell_type"] == cell_type_names[args.cell_type]
        y_test_pred[~s] = np.nan
    print("save results")
    if args.task_type == "multi":
        pred_file_path = "multimodal_pred.pickle"
    elif args.task_type == "cite":
        pred_file_path = "citeseq_pred.pickle"
    else:
        raise ValueError
    with open(os.path.join(args.out_dir, pred_file_path), "wb") as f:
        pickle.dump(y_test_pred, f)
print("completed !")



# if __name__ == "__main__":
#     main()

#####################

import importlib
# 1) 先重载定义 LatentModel 的模块
import ss_opm.model.encoder_decoder.mlp_module as mlp_mod
print("mlp_module @", mlp_mod.__file__)           # 确认路径就是你正在编辑的那个
importlib.reload(mlp_mod)
print("has LatentModel?", hasattr(mlp_mod, "LatentModel"))

import ss_opm.model.encoder_decoder.encoder_decoder as encdec_mod

encdec_mod = importlib.reload(encdec_mod)   # 重新加载模块本体
EncoderDecoder = encdec_mod.EncoderDecoder  # 重新拿到最新的类引用

import numpy as np
import torch
from types import SimpleNamespace
import numpy as np
import torch
import importlib, types

# ========= 1) 读数据（按你脚本的接口） =========
from ss_opm.utility.load_dataset import load_dataset
from ss_opm.pre_post_processing.pre_post_processing import PrePostProcessing
from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
task_type = "cite"   # 或 "multi"
data_dir  = r"D:\04_project\03_RNA2ADT\data\processed"

train_inputs, train_metadata, train_target = load_dataset(
    data_dir=data_dir, task_type=task_type, split="train", cell_type="all"
)

ts = np.load(r'D:\04_project\03_RNA2ADT\data\processed\train_cite_inputs_idxcol.npz', allow_pickle=True)

print(ts.files)  # ['index', 'columns']

idx = ts['index']     # 可能是 dtype=object 的 cell IDs
cols = ts['columns']  # 可能是 dtype=object 的 gene/protein 名
genes = np.char.partition(cols.astype('U'), '_')[:, 0]
import sys
sys.path.append(r'D:\04_project\03_RNA2ADT')
# from myplm.embedder.omics_cross_esm import OmicsEmbeddingLayer
def load_gene_embeddings(gene_emb_path, pretrain_gene_list):
    """加载基因embeddings，只读取一次"""
    if gene_emb_path is None:
        gene_emb_path = r'D:\02_bioinformatics\04_st_imputaiton\scPRINT\data\main\gene_embeddings.parquet'
    
    print(f"Loading gene embeddings from {gene_emb_path}...")
    all_embeddings = pd.read_parquet(gene_emb_path)
    
    # 检查哪些基因在embedding文件中可用
    available_genes = [gene for gene in pretrain_gene_list if gene in all_embeddings.index]
    missing_genes = [gene for gene in pretrain_gene_list if gene not in all_embeddings.index]
    
    if len(available_genes) == 0:
        raise ValueError(
            f"the gene embeddings file {gene_emb_path} does not contain any of the genes given to the model"
        )
    elif len(available_genes) < len(pretrain_gene_list):
        print(
            "Warning: only a subset of the genes available in the embeddings file."
        )
        print("number of genes: ", len(available_genes))
    
    if len(missing_genes) > 0:
        print(f"Warning: {len(missing_genes)} genes not found in ESM2 embeddings, will use random initialization for them")
        print(f"Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"Missing genes: {missing_genes}")
    
    # 提取可用的embeddings
    available_embeddings = all_embeddings.loc[available_genes]
    
    print(f"Successfully loaded {len(available_genes)} gene embeddings")
    return available_embeddings, available_genes, missing_genes

gene_embeddings_data = load_gene_embeddings(r'D:\02_bioinformatics\04_st_imputaiton\scPRINT\data\main\gene_embeddings.parquet', genes)


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hid, gene_emb=None, fix_embedding=False, 
                precpt_gene_emb=None, gene_embeddings_data=None):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = dict(zip(pretrained_gene_list, list(range(len(pretrained_gene_list)))))
        self.num_hid = num_hid

        # Handle ESM2 embeddings - prefer pre-loaded data over file path
        if gene_embeddings_data is not None:
            # Use pre-loaded embeddings data
            available_embeddings, available_genes, missing_genes = gene_embeddings_data
            
            if len(available_genes) == 0:
                raise ValueError(
                    f"the pre-loaded gene embeddings do not contain any of the genes given to the model"
                )
            elif len(available_genes) < len(pretrained_gene_list):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(available_genes))
            
            if len(missing_genes) > 0:
                print(f"Warning: {len(missing_genes)} genes not found in ESM2 embeddings, will use random initialization for them")
                print(f"Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"Missing genes: {missing_genes}")
            
            # Initialize embeddings tensor with random values
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            
            # Fill in ESM2 embeddings for available genes
            if len(available_genes) > 0:
                sembeddings = torch.nn.AdaptiveAvgPool1d(num_hid)(
                    torch.tensor(available_embeddings.values, dtype=torch.float32)
                )
                
                # Map available genes to their positions in pretrained_gene_list
                for i, gene in enumerate(pretrained_gene_list):
                    if gene in available_genes:
                        gene_idx_in_available = available_genes.index(gene)
                        self.emb.data[i] = sembeddings[gene_idx_in_available]
            
            # Create mask for which embeddings should be frozen (ESM2 embeddings)
            self.esm2_mask = torch.zeros(len(pretrained_gene_list), dtype=torch.bool)
            for i, gene in enumerate(pretrained_gene_list):
                if gene in available_genes:
                    self.esm2_mask[i] = True
            
            if fix_embedding:
                # Only freeze ESM2 embeddings, allow missing genes to be trained
                self.emb.requires_grad = True
                # We'll handle freezing in forward pass or optimizer step
                
        elif precpt_gene_emb is not None:
            # Fallback to loading from file (original behavior)
            if precpt_gene_emb is None:
                precpt_gene_emb = '/l/users/yu.li/zgy/scPRINT/data/main/gene_embeddings.parquet'
            
            # Load all embeddings from parquet file
            all_embeddings = pd.read_parquet(precpt_gene_emb)
            
            # Check which genes are available in the embedding file
            available_genes = [gene for gene in pretrained_gene_list if gene in all_embeddings.index]
            missing_genes = [gene for gene in pretrained_gene_list if gene not in all_embeddings.index]
            
            if len(available_genes) == 0:
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(available_genes) < len(pretrained_gene_list):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(available_genes))
            
            if len(missing_genes) > 0:
                print(f"Warning: {len(missing_genes)} genes not found in ESM2 embeddings, will use random initialization for them")
                print(f"Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"Missing genes: {missing_genes}")
            
            # Initialize embeddings tensor with random values
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            
            # Fill in ESM2 embeddings for available genes
            if len(available_genes) > 0:
                available_embeddings = all_embeddings.loc[available_genes]
                sembeddings = torch.nn.AdaptiveAvgPool1d(num_hid)(
                    torch.tensor(available_embeddings.values, dtype=torch.float32)
                )
                
                # Map available genes to their positions in pretrained_gene_list
                for i, gene in enumerate(pretrained_gene_list):
                    if gene in available_genes:
                        gene_idx_in_available = available_genes.index(gene)
                        self.emb.data[i] = sembeddings[gene_idx_in_available]
            
            # Create mask for which embeddings should be frozen (ESM2 embeddings)
            self.esm2_mask = torch.zeros(len(pretrained_gene_list), dtype=torch.bool)
            for i, gene in enumerate(pretrained_gene_list):
                if gene in available_genes:
                    self.esm2_mask[i] = True
            
            if fix_embedding:
                # Only freeze ESM2 embeddings, allow missing genes to be trained
                self.emb.requires_grad = True
                # We'll handle freezing in forward pass or optimizer step
                
        elif gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
            self.esm2_mask = None
        else:
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            self.esm2_mask = None
            if fix_embedding:
                self.emb.requires_grad = False
        self._esm_hook_handle = None

    def freeze_esm2_embeddings(self):
        """Freeze only ESM2 embeddings while allowing others to be trained."""
        # 已注册过就别再注册
        if getattr(self, "_esm_hook_handle", None) is not None:
            return
        if hasattr(self, 'esm2_mask') and self.esm2_mask is not None:
            mask = self.esm2_mask.to(self.emb.device)  # 确保同device
            def hook(grad):
                grad = grad.clone()            # 避免原地修改
                grad[mask] = 0                 # 冻结 ESM2 对应行
                return grad
            self._esm_hook_handle = self.emb.register_hook(hook)

    def forward(self, x_dict, input_gene_list=None):

        x = x_dict
        # if 'dropout' in x_dict:
        #     indices = x._indices().t()
        #     values = x._values()
        #     temp = values.sum()
        #     values = values.float()
        #     values = torch.distributions.binomial.Binomial(values, x_dict['dropout']).sample()
        #     x = torch.sparse.FloatTensor(indices.t(), values, x.shape)

        # x = torch.log1p(x)
        # x = sparse_tpm(x)
        if input_gene_list is not None:
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()
            # x_dict['input_gene_mask'] = gene_idx
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError('The input gene size is not the same as the pretrained gene list. Please provide the input gene list.')
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)
        feat = F.embedding(gene_idx, self.emb)
        feat = torch.sparse.mm(x, feat)

        gene_emb = x.to_dense().unsqueeze(-1) * self.emb[gene_idx].unsqueeze(0)

        return feat, gene_emb

class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, gene_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, gene_emb=None,
                 inject_covariate=False, batch_num=None, precpt_gene_emb=None, freeze_embeddings=False, gene_embeddings_data=None):
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()#create_activation(activation)
        self.norm0 = nn.LayerNorm(num_hidden) #create_norm(norm, num_hidden) #nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            # create_norm(norm, num_hidden),
            nn.LayerNorm(num_hidden)
        )
        if pe_type is not None:
            if cat_pe:
                num_emb = num_hidden // 2
            else:
                num_emb = num_hidden
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        if gene_emb is None and precpt_gene_emb is None and gene_embeddings_data is None:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb)
        else:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb, gene_emb, freeze_embeddings, precpt_gene_emb, gene_embeddings_data)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False
        # OmicsEmbeddingLayer.__init__ 里，创建完 self.feat_enc 之后加
        if hasattr(self.feat_enc, 'esm2_mask') and self.feat_enc.esm2_mask is not None:
            self.feat_enc.freeze_esm2_embeddings()  # 只在构造时注册一次

    def forward(self, x_dict, input_gene_list=None):
        # Apply gradient masking if using ESM2 embeddings
        # if hasattr(self.feat_enc, 'esm2_mask') and self.feat_enc.esm2_mask is not None:
        #     self.feat_enc.freeze_esm2_embeddings()
        
        x, gene_emb = self.feat_enc(x_dict, input_gene_list)#self.act(self.feat_enc(x_dict, input_gene_list))

        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]
            pe = self.pe_enc(pe_input) #0.
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict['batch'])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)

        return gene_emb, x


embedder = OmicsEmbeddingLayer(gene_list=genes, num_hidden = 64, norm='layernorm', activation='gelu',
                                dropout=0.2, pe_type=None, cat_pe=True, gene_emb=None, gene_embeddings_data=gene_embeddings_data,
                                inject_covariate=False, batch_num=None)
embedder.to(device=device)

gene_to_idx = {g: i for i, g in enumerate(genes)}   # 或者用全量基因表建好映射
unk_id = 0  # 可选，用来处理未知基因

gene_idx = torch.tensor(
    [gene_to_idx.get(g, unk_id) for g in genes],
    dtype=torch.long,
    device=next(embedder.parameters()).device
)
device = 'cuda:0'
x = train_inputs[:10,].toarray()
x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
gene_emb, x = embedder(x, genes.tolist())
x.shape


class GeneFlowAttention(nn.Module):
    """专门用于基因自注意力的 Flow Attention 实现"""
    def __init__(self, d_model, n_heads, drop_out=0.01, eps=1e-6):
        super(GeneFlowAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, x):
        # input: (B, L, D); output: (B, L, D)
        B, L, _ = x.shape
        
        # 1. Linear projection
        queries = self.query_projection(x).view(B, L, self.n_heads, self.head_dim)
        keys = self.key_projection(x).view(B, L, self.n_heads, self.head_dim)
        values = self.value_projection(x).view(B, L, self.n_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)  # (B, n_heads, L, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        
        # 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        
        # (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


class GeneFlowformerLayer(nn.Module):
    """专门用于基因自注意力的 Flowformer Layer"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, norm='layernorm', norm_first=True):
        super(GeneFlowformerLayer, self).__init__()
        self.self_attn = GeneFlowAttention(embed_dim, num_heads, dropout)
        self._ff_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim) if norm == 'layernorm' else nn.BatchNorm1d(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if norm == 'layernorm' else nn.BatchNorm1d(embed_dim)
        self.norm_first = norm_first

    def forward(self, x, attn_mask=None, output_attentions=False):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x):
        # x 已经是 (B, L, D) 格式，直接传递给 GeneFlowAttention
        x = self.self_attn(x)
        return self.dropout1(x)


class Flowenc(nn.Module):
    """
    将可学习的 CLS（cell embedding）添加到 gene embedding 前，再做 self-attention（无注意力掩码）
    """
    def __init__(self, d: int = 1024, heads: int = 4, nlayers: int = 2, 
                 dropout: float = 0.1, cell_emb_style: str = "cls", cross_attn: bool = False):
        super().__init__()
        if cell_emb_style not in ["cls", "mean"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        self.cell_emb_style = cell_emb_style
        self.d = d
        self.heads = heads

        head_dim = d // heads
        if head_dim * heads != d:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads}).")

        # Layer Norms
        self.ln_g = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)

        # Gene+Cell 的 self-attention（沿用你已有的 GeneFlowformerLayer 定义）
        self.flowformer = nn.Sequential(*[
            GeneFlowformerLayer(embed_dim=d, num_heads=heads, dropout=dropout, norm='layernorm')
            for _ in range(nlayers)
        ])

        # ===== 新增：可学习 CLS token =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # 典型初始化
        self.dp = nn.Dropout(dropout)

    def forward(
        self,
        gene_tok: torch.Tensor,      # (B, G, d)
        cell_tok: torch.Tensor = None,  # (B, d) 或 None
        *,
        use_pool_cell: bool = True,     # 是否采用 pooling 生成 cell embedding（仅当 cell_emb_style=="mean" 时生效）
        weights: torch.Tensor = None    # w-pool 的权重 (B, G) 或 (B, G, 1)
    ):
        B, G, d = gene_tok.shape
        device = gene_tok.device

        # === 生成 cell_src ===
        if self.cell_emb_style == "cls":
            # 使用可学习的 CLS；若你传入了 cell_tok 想覆盖，也可以替换为:
            # cell_src = cell_tok if (cell_tok is not None) else self.cls_token.expand(B, 1, d).squeeze(1)
            cell_src = self.cls_token.expand(B, 1, d).squeeze(1)  # (B, d)
        else:
            # mean pooling（无 mask；若给了 weights 则做加权均值）
            if (not use_pool_cell) and (cell_tok is not None):
                cell_src = cell_tok  # 显式传入时直接用
            else:
                if weights is None:
                    cell_src = gene_tok.mean(dim=1)  # (B, d)
                else:
                    if weights.dim() == 2:
                        weights = weights.unsqueeze(-1)  # (B,G,1)
                    w = weights / (weights.sum(dim=1, keepdim=True).clamp(min=1e-6))
                    cell_src = (gene_tok * w).sum(dim=1)  # (B, d)

        # 拼接 CLS 到最前： (B, G+1, d)
        cell_cls = cell_src.unsqueeze(1)                 
        gene_cell_tok = torch.cat([cell_cls, gene_tok], dim=1)

        # 无掩码，直接 LN + Transformer
        g_norm = self.ln_g(gene_cell_tok)
        gene_cell_out = self.flowformer(g_norm)
        gene_cell_tok = gene_cell_tok + self.dp(gene_cell_out)

        # 拆回
        cell_tok_out = gene_cell_tok[:, 0, :]    # (B, d)

        return cell_tok_out

encoder = Flowenc(d=64, heads=4, nlayers=2, dropout=0.2, cell_emb_style="cls")
encoder.to(device=device)
h = encoder(gene_emb, x)  # x: (B, G, d) 格式
h.shape


# ts = D:\04_project\03_RNA2ADT\open-problems-multimodal\result\citeseq_pred.pickle
import pickle
import numpy as np
import pandas as pd

# 1. 读取你的 CITE 预测
ts = r"D:\04_project\03_RNA2ADT\open-problems-multimodal\result\citeseq_pred.pickle"

with open(ts, "rb") as f:
    pred = pickle.load(f)

print("pred shape:", pred.shape)  # 应该是 (48663, 140)

# 2. 展平成一维向量（按行展开）
cite_flat = pred.reshape(-1)
print("CITE flat length:", cite_flat.shape[0])  # 应该是 6812820

# 3. Kaggle 总共需要的行数
TOTAL_ROWS = 65744180
CITE_ROWS = cite_flat.shape[0]
MULTI_ROWS = TOTAL_ROWS - CITE_ROWS
print("Need MULTI rows:", MULTI_ROWS)  # 应该是 58931360

# 4. Multiome 部分先用 0 填充（占位）
multi_flat = np.zeros(MULTI_ROWS, dtype=np.float32)

# 5. 拼接 CITE + Multiome
all_target = np.concatenate([cite_flat, multi_flat])
assert all_target.shape[0] == TOTAL_ROWS

# 6. 生成提交用的 DataFrame
df = pd.DataFrame({
    "row_id": np.arange(TOTAL_ROWS, dtype=np.int64),
    "target": all_target
})

# 7. 保存为压缩 CSV（体积小一点，Kaggle 支持）
df.to_csv("submission.csv.gz", index=False)
print("Saved to submission.csv.gz")


import numpy as np
import pandas as pd
import pickle

# 1. 读你的 CITE 预测矩阵 (48663, 140)
with open(r"D:\04_project\03_RNA2ADT\open-problems-multimodal\result\citeseq_pred.pickle", "rb") as f:
    pred = pickle.load(f)   # shape: (n_cells, n_proteins)

# 2. 读 CITE 的 test 输入，拿到 cell_id 顺序
#    你之前已经能读 train_cite_inputs.h5，那 test_cite_inputs.h5 也一样
import h5py

with h5py.File(r"D:\04_project\03_RNA2ADT\open-problems-multimodal\test_cite_inputs.h5", "r") as f:
    cell_ids = f['axis0'][:]      # 或实际 key，看你本地；通常是 bytes，要 decode 一下
    cell_ids = cell_ids.astype(str)

# 3. 读 CITE 的 target，拿到 protein / ADT 列顺序
with h5py.File(r"D:\04_project\03_RNA2ADT\open-problems-multimodal\train_cite_targets.h5", "r") as f:
    protein_ids = f['axis1'][:]   # 列名
    protein_ids = protein_ids.astype(str)

# 4. 建立 id -> index 映射
cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
prot_id_to_idx = {pid: j for j, pid in enumerate(protein_ids)}

# 5. 读 evaluation_ids（Kaggle 数据目录里的）
eval_df = pd.read_csv(r"D:\04_project\03_RNA2ADT\open-problems-multimodal\evaluation_ids.csv")

# 假设 eval_df 里有: row_id, cell_id, gene_id 三列
# （列名可能是 'gene_id' 或 'feature_id'，你对应改一下）

N = len(eval_df)
targets = np.zeros(N, dtype=np.float32)

# 6. 逐行填 CITE 的预测
for i, row in eval_df.iterrows():
    cid = row["cell_id"]
    gid = row["gene_id"]      # 如果列叫别的，改这里

    if gid in prot_id_to_idx:  # 说明这是 CITE 那边的一个蛋白
        ci = cell_id_to_idx[cid]
        gi = prot_id_to_idx[gid]
        targets[i] = pred[ci, gi]   # 从你的 (48663,140) 里取值
    else:
        # Multi 那部分：暂时先填 0（或者可以用 sample_submission 的默认值）
        targets[i] = 0.0

# 7. 构造 submission DataFrame（注意 row_id 顺序）
sub = pd.DataFrame({
    "row_id": eval_df["row_id"].values,   # 确保跟 evaluation_ids 对齐
    "target": targets,
})

sub.to_csv("submission.csv.gz", index=False)
print("saved submission.csv.gz")

########
import os
import pickle
import numpy as np
import pandas as pd

# ==== 路径自行改成你的 ====
base = r"D:\04_project\03_RNA2ADT\open-problems-multimodal"
proc = os.path.join(r"D:\04_project\03_RNA2ADT", "data", "processed")

# 1. 读取你训练好的 CITE 预测 (48663, 140)
with open(os.path.join(base, "result", "citeseq_pred.pickle"), "rb") as f:
    pred = pickle.load(f)   # shape: (n_cells, n_proteins)

print("pred shape:", pred.shape)

# 2. 读取 CITE test 的 cell 顺序
#    test_cite_inputs_idxcol.npz 里一般有 'index' 和 'columns'
test_idx = np.load(os.path.join(proc, "test_cite_inputs_idxcol.npz"),
                   allow_pickle=True)
print("test_cite_inputs_idxcol keys:", test_idx.files)

# 通常 cell 都在 'index' 里，如果你跑出来不是，就把 key 换掉
cell_ids = test_idx["index"].astype(str)

# 3. 读取 CITE target 的 protein / ADT 列顺序
tgt_idx = np.load(os.path.join(proc, "train_cite_targets_idxcol.npz"),
                  allow_pickle=True)
print("train_cite_targets_idxcol keys:", tgt_idx.files)

# 一般 protein / ADT 名在 'columns' 里，有些脚本会叫 'index'，不对就调一下
protein_ids = tgt_idx["columns"].astype(str)

print("n_cells =", len(cell_ids), "n_proteins =", len(protein_ids))

# 4. 建立 id → 下标 的映射
cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
prot_id_to_idx = {pid: j for j, pid in enumerate(protein_ids)}

# 5. 读 evaluation_ids 和 sample_submission
eval_df = pd.read_parquet(os.path.join(proc, "evaluation_ids.parquet"))
sub = pd.read_parquet(os.path.join(proc, "sample_submission.parquet"))

print("evaluation_ids columns:", eval_df.columns)

# 有的脚本叫 gene_id，有的叫 feature_id，这里做个兼容
if "gene_id" in eval_df.columns:
    feat_col = "gene_id"
elif "feature_id" in eval_df.columns:
    feat_col = "feature_id"
else:
    raise ValueError("找不到基因/蛋白列，检查 evaluation_ids.parquet 的列名。")

if "cell_id" not in eval_df.columns:
    raise ValueError("找不到 cell_id 列，检查 evaluation_ids.parquet 的列名。")

# 6. 先把 sample_submission 的 target 拿出来，当默认值
targets = sub["target"].to_numpy(dtype=np.float32)
row_ids = eval_df["row_id"].to_numpy()

# 7. 只对 CITE 的那些行更新：gene 在 protein_ids 里面的就是 CITE
is_cite = eval_df[feat_col].isin(protein_ids).to_numpy()

cite_cells  = eval_df.loc[is_cite, "cell_id"].to_numpy()
cite_genes  = eval_df.loc[is_cite, feat_col].to_numpy()
cite_rowids = eval_df.loc[is_cite, "row_id"].to_numpy()

print("num CITE rows in evaluation_ids:", cite_rowids.shape[0])

# 8. 把你的 (n_cells, n_proteins) 预测按 row_id 写回去
for cid, gid, rid in zip(cite_cells, cite_genes, cite_rowids):
    ci = cell_id_to_idx[cid]
    gi = prot_id_to_idx[gid]
    targets[rid] = pred[ci, gi]

# 9. 回写到 submission，并保存成 csv.gz
sub["target"] = targets
out_path = os.path.join(base, "submission.csv.gz")
sub.to_csv(out_path, index=False)
print("Saved submission to:", out_path)
