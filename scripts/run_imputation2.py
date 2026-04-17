"""
run_imputation2.py"""

import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool


def has_graph_support(model_cls):
    return model_cls in [models.GRINet]


def get_model_classes(model_str):
    if model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name):
    if dataset_name == 'manglaria':
    # ── CARGA Y PREPARACIÓN ───────────────────────────────────────────────────
        DATA_PATH     = 'manglaria_abril_timeseries.csv'
        TIMESTAMP_COL = 'TIMESTAMP'
        TIMESTAMP_FMT = '%d/%m/%Y %H:%M'
    
        df_raw = pd.read_csv(DATA_PATH)
        df_raw[TIMESTAMP_COL] = pd.to_datetime(
            df_raw[TIMESTAMP_COL], format=TIMESTAMP_FMT
        )
        df_raw = df_raw.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        df_raw = df_raw.set_index(TIMESTAMP_COL)
    
        # Solo columnas numéricas — excluye TIMESTAMP automáticamente
        df_num = df_raw.select_dtypes(include=[np.number])
    
        # ── CLASE ADAPTADOR ───────────────────────────────────────────────────────
        class ManglarIAAdapter:
            def __init__(self, df):
                import numpy as np
                import pandas as pd

                T, N = df.shape

                # ── MÁSCARA ───────────────────────────────────────────────────
                mask_np         = (~df.isnull()).values.astype(np.uint8)
                self._mask_full = mask_np.reshape(T, N, 1)

                # ── FILL + NORMALIZACIÓN ──────────────────────────────────────
                df_filled    = df.ffill().bfill().fillna(0.0)
                n_train      = int(T * 0.7)
                self._mean   = df_filled.iloc[:n_train].mean()
                self._std    = df_filled.iloc[:n_train].std().replace(0, 1.0)
                df_norm      = (df_filled - self._mean) / self._std
                self._data   = df_norm.values.reshape(T, N, 1)

               # ── EVAL MASK ─────────────────────────────────────────────────────
                mask_path = os.environ.get('GRIN_MASK_PATH', '')
                print(f"DEBUG [Adapter]: GRIN_MASK_PATH = {mask_path}")

                # 1. Huecos reales del dataset
                real_mask = df.isnull().values.astype(np.uint8)  # (T, N)
                print(f"DEBUG [Adapter]: real_mask shape={real_mask.shape}, missing%={real_mask.mean()*100:.2f}%")

                # 2. Máscara artificial (si se proporcionó)
                if mask_path and os.path.exists(mask_path):
                    artificial = np.load(mask_path).astype(np.uint8)
                    print(f"DEBUG [Adapter]: artificial mask loaded, shape={artificial.shape}, missing%={artificial.mean()*100:.2f}%")
                    # Unión: un hueco es missing si es real O artificial
                    eval_mask_np = np.logical_or(real_mask, artificial).astype(np.uint8)
                    print(f"DEBUG [Adapter]: combined mask missing%={eval_mask_np.mean()*100:.2f}%")
                else:
                    print("DEBUG [Adapter]: No external mask, using only real missing")
                    eval_mask_np = real_mask

                self._eval_mask = eval_mask_np.reshape(T, N, 1)
                print(f"DEBUG [Adapter]: final eval_mask shape={self._eval_mask.shape}")

                # ── ÍNDICE ────────────────────────────────────────────────────
                self._index = np.arange(T)

                # ── ADYACENCIA ────────────────────────────────────────────────
                adj = np.ones((N, N), dtype=float)
                np.fill_diagonal(adj, 0.0)
                self._adj = adj

                # ── DATAFRAME ─────────────────────────────────────────────────
                self.df = df_norm
                self._N = N
                self._T = T
    
            def numpy(self, return_idx=False):
                if return_idx:
                    return self._data, self._index
                return self._data
    
            @property
            def training_mask(self):
                # Datos visibles durante entrenamiento:
                # presentes en el CSV menos los reservados para evaluación
                # Con tan pocos huecos reales (0.2%) training_mask ≈ mask_full
                return self._mask_full & (1 - self._eval_mask)
    
            @property
            def eval_mask(self):
                # Huecos reales del CSV — el modelo los intentará imputar
                return self._eval_mask
    
            def splitter(self, dataset, val_len=0.1, test_len=0.2):
                import numpy as np
                n       = len(dataset)
                n_test  = int(n * test_len)
                n_val   = int(n * val_len)
                n_train = n - n_test - n_val
                # División cronológica: train primero, test al final
                # Importante para series temporales — no mezclar futuro con pasado
                return [
                    np.arange(0, n_train),
                    np.arange(n_train, n_train + n_val),
                    np.arange(n_train + n_val, n)
                ]
    
            def get_similarity(self, thr=0., sparse=False, **kwargs):
                import numpy as np
                adj = self._adj.copy()
                adj[adj < thr] = 0.
                return adj
    
        return ManglarIAAdapter(df_num)
    elif dataset_name == 'mexflux':

        DATA_PATH     = 'mexflux.csv'
        TIMESTAMP_COL = 'timestamp'
        TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%f UTC'
        SITE_COL      = 'site_id'

        # Columnas a excluir — no son variables de medición
        EXCLUIR = ['primary_key', 'timestamp', 'DOY', 'site_id']

        df_raw = pd.read_csv(DATA_PATH)
        df_raw[TIMESTAMP_COL] = pd.to_datetime(
            df_raw[TIMESTAMP_COL], format=TIMESTAMP_FMT, utc=True
        )

        # Separar por sensor
        sensores = sorted(df_raw[SITE_COL].unique())  # ['RBMNN', 'RBRL']
        var_cols  = [c for c in df_raw.columns if c not in EXCLUIR]

        # Construir índice temporal unificado con todos los timestamps
        # de ambos sensores combinados y ordenados
        todos_timestamps = pd.DatetimeIndex(
            pd.concat([df_raw[df_raw[SITE_COL] == s][TIMESTAMP_COL]
                    for s in sensores])
        ).unique().sort_values()

        # Crear DataFrame por sensor con el índice unificado
        # Las filas donde el sensor no tiene medición quedan como NaN
        dfs = {}
        for s in sensores:
            sub = df_raw[df_raw[SITE_COL] == s].copy()

            # ── ELIMINAR DUPLICADOS POR TIMESTAMP ───────────────
            sub = sub.sort_values(TIMESTAMP_COL)
            sub = sub.drop_duplicates(subset=[TIMESTAMP_COL], keep='last')

            sub = sub.set_index(TIMESTAMP_COL)
            sub = sub[var_cols]

            dfs[s] = sub.reindex(todos_timestamps)

        class MexFluxAdapter:
            def __init__(self, dfs, sensores, todos_timestamps):
                import numpy as np
                import pandas as pd

                T  = len(todos_timestamps)
                N  = len(sensores)       # 2 sensores = 2 nodos
                d  = len(var_cols)       # 23 variables

                # ── DATOS (T, N, d) ───────────────────────────────────────────
                # Apilamos los dos sensores en la dimensión N
                datos_raw = np.stack(
                    [dfs[s].values for s in sensores], axis=1
                )  # (T, N, d)

                # ── MÁSCARA ───────────────────────────────────────────────────
                # 1 = dato presente, 0 = faltante (real o por ausencia del sensor)
                mask_np = np.stack(
                    [(~dfs[s].isnull()).values for s in sensores], axis=1
                ).astype(np.uint8)  # (T, N, d)
                self._mask_full = mask_np

                # ── FILL + NORMALIZACIÓN ──────────────────────────────────────
                # Primero llenamos NaN para poder normalizar
                datos_filled = np.copy(datos_raw)
                for n in range(N):
                    df_tmp = pd.DataFrame(datos_raw[:, n, :])
                    df_tmp = df_tmp.ffill().bfill().fillna(0.0)
                    datos_filled[:, n, :] = df_tmp.values

                # Normalización z-score sobre el 70% de train
                # Calculamos media y std sobre todos los sensores combinados
                n_train = int(T * 0.7)
                datos_train = datos_filled[:n_train]  # (n_train, N, d)

                # Media y std por variable (promediando sobre T y N)
                self._mean = datos_train.reshape(-1, d).mean(axis=0)  # (d,)
                self._std  = datos_train.reshape(-1, d).std(axis=0)   # (d,)
                self._std[self._std == 0] = 1.0  # evitar división por cero

                datos_norm = (datos_filled - self._mean) / self._std  # (T, N, d)
                self._data  = datos_norm

                # ── EVAL MASK ─────────────────────────────────────────────────
                # Huecos reales = donde el sensor no tenía medición
                eval_mask_np = (1 - mask_np).astype(np.uint8)  # (T, N, d)
                self._eval_mask = eval_mask_np

                # ── ÍNDICE ────────────────────────────────────────────────────
                self._index = np.arange(T)

                # ── ADYACENCIA ────────────────────────────────────────────────
                # Grafo totalmente conectado entre los 2 sensores
                adj = np.ones((N, N), dtype=float)
                np.fill_diagonal(adj, 0.0)
                self._adj = adj

                # ── DATAFRAME para análisis post-entrenamiento ─────────────────
                # Aplanamos (T, N, d) → (T, N*d) para compatibilidad
                self.df = pd.DataFrame(
                    datos_norm.reshape(T, N * d),
                    index=todos_timestamps
                )
                self._N = N
                self._T = T
                self._d = d

                print(f"MexFlux cargado: T={T}, N={N} sensores, d={d} variables")
                print(f"  Sensores: {sensores}")
                for i, s in enumerate(sensores):
                    n_presentes = mask_np[:, i, 0].sum()
                    pct = n_presentes / T * 100
                    print(f"  {s}: {n_presentes}/{T} pasos con datos ({pct:.1f}%)")

            def numpy(self, return_idx=False):
                if return_idx:
                    return self._data, self._index
                return self._data

            @property
            def training_mask(self):
                return self._mask_full & (1 - self._eval_mask)

            @property
            def eval_mask(self):
                return self._eval_mask

            def splitter(self, dataset, val_len=0.1, test_len=0.2):
                import numpy as np
                n       = len(dataset)
                n_test  = int(n * test_len)
                n_val   = int(n * val_len)
                n_train = n - n_test - n_val
                return [
                    np.arange(0, n_train),
                    np.arange(n_train, n_train + n_val),
                    np.arange(n_train + n_val, n)
                ]

            def get_similarity(self, thr=0., sparse=False, **kwargs):
                import numpy as np
                adj = self._adj.copy()
                adj[adj < thr] = 0.
                return adj

        return MexFluxAdapter(dfs, sensores, todos_timestamps)
    elif dataset_name == 'synthetic':
        from lib.datasets.synthetic import ChargedParticles
        
        # Cargamos el dataset original sin tocarlo
        raw = ChargedParticles(
            static_adj=False,
            p_block=float(os.environ.get('GRIN_P_BLOCK', '0.025')),
            p_point=float(os.environ.get('GRIN_P_POINT', '0.025')),
            min_seq=int(os.environ.get('GRIN_MIN_SEQ', '4')),
            max_seq=int(os.environ.get('GRIN_MAX_SEQ', '9')),
            use_exogenous=False
        )
        
        class SyntheticAdapter:
            def __init__(self, raw):
                import numpy as np
                import pandas as pd
                
                # ── DATOS ──────────────────────────────────────────
                # raw.loc: (5000, 50, 10, 2)
                # n_sims=5000 simulaciones, T=50 pasos, N=10 nodos, d=2 dims
                loc      = raw.loc.numpy()
                mask     = raw.mask.numpy()
                eval_m   = raw.eval_mask.numpy()  # original, sin modificar
                n_sims, T, N, d = loc.shape
                
                # Concatenamos simulaciones y aplanamos nodos+dims
                # (5000*50, 10*2) = (250000, 20)
                self._data      = loc.reshape(n_sims * T, N, d)
                self._mask      = mask.reshape(n_sims * T, N, d)
                self._eval_mask = eval_m.reshape(n_sims * T, N, d)
                self._index     = np.arange(n_sims * T)
                
                # ── ADYACENCIA ─────────────────────────────────────
                # ChargedParticles.get_similarity devuelve matriz totalmente
                # conectada (10,10) sin self-loops — perfecta para empezar
                self._adj = raw.get_similarity()

                # Creamos un DataFrame simple con los datos aplanados
                # para que el script pueda hacer sus análisis post-entrenamiento
                self.df = pd.DataFrame(
                    self._data.reshape(len(self._data), -1)
                )
            
            def numpy(self, return_idx=False):
                # Lo que run_imputation usa para construir ImputationDataset
                if return_idx:
                    return self._data, self._index
                return self._data
            
            @property
            def training_mask(self):
                # Datos visibles durante entrenamiento:
                # mask=1 donde hay valor, MENOS los huecos de evaluación
                return self._mask & (1 - self._eval_mask)
            
            @property
            def eval_mask(self):
                # Huecos artificiales para medir qué tan bien imputa GRIN
                return self._eval_mask
            
            def splitter(self, dataset, val_len=0.1, test_len=0.2):
                import numpy as np
                n       = len(dataset)
                n_test  = int(n * test_len)
                n_val   = int(n * val_len)
                n_train = n - n_test - n_val
                return [
                    np.arange(0, n_train),
                    np.arange(n_train, n_train + n_val),
                    np.arange(n_train + n_val, n)
                ]
            
            def get_similarity(self, thr=0., sparse=False, **kwargs):
                import numpy as np
                adj = self._adj.copy().astype(float)
                adj[adj < thr] = 0.
                return adj
    
        return SyntheticAdapter(raw)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='air36')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # === DEBUG: Verificar máscara externa ===
    mask_path_env = os.environ.get('GRIN_MASK_PATH', '')
    print(f"DEBUG: GRIN_MASK_PATH = {mask_path_env}")

    model_cls, filler_cls = get_model_classes(args.model_name)
    print("DEBUG: Creando dataset...")
    dataset = get_dataset(args.dataset_name)
    print("DEBUG: Dataset creado")

     # === DEBUG: Verificar máscara final del dataset ===
    if hasattr(dataset, 'eval_mask'):
        eval_mask = dataset.eval_mask
        print(f"DEBUG: dataset.eval_mask shape={eval_mask.shape}, missing%={eval_mask.mean()*100:.2f}%")
    else:
        print("DEBUG: dataset no tiene eval_mask")

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)
    print(f"DEBUG: train_idxs length={len(train_idxs)}, val_idxs={len(val_idxs)}, test_idxs={len(test_idxs)}")

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # === DEBUG: Verificar slices del DataModule ===
    print(f"DEBUG: dm.train_slice = {dm.train_slice}")
    print(f"DEBUG: dm.val_slice = {dm.val_slice}")
    print(f"DEBUG: dm.test_slice = {dm.test_slice}")
    print(f"DEBUG: test_slice length = {len(dm.test_slice) if dm.test_slice is not None else 0}")
    if dm.test_slice is not None and len(dm.test_slice) == 0:
        print("ERROR: test_slice vacío, no hay datos para evaluar")
        # Opcional: salir o retornar algo vacío

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     })
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    print("DEBUG: Iniciando entrenamiento...")
    trainer.fit(filler, datamodule=dm)
    print("DEBUG: Entrenamiento completado")

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    print("DEBUG: Iniciando testing (trainer.test)...")
    trainer.test()
    print("DEBUG: trainer.test completado")
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()
    
    print("DEBUG: Ejecutando predict_loader...")
    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)
    print("DEBUG: predict_loader completado")
    print(f"DEBUG: y_true shape (tensor): {y_true.shape}, y_hat: {y_hat.shape}, mask: {mask.shape}")    
    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[0], y_hat.shape[1], -1)
    print(f"DEBUG: y_hat convertido a numpy, shape={y_hat.shape}")

    # Test imputations in whole series
    try:
        eval_mask = dataset.eval_mask[dm.test_slice]
        df_true = dataset.df.iloc[dm.test_slice]
        metrics = {
            'mae': numpy_metrics.masked_mae,
            'mse': numpy_metrics.masked_mse,
            'mre': numpy_metrics.masked_mre,
            'mape': numpy_metrics.masked_mape
        }
        index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
        aggr_methods = ensure_list(args.aggregate_by)
        df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
        df_hats = dict(zip(aggr_methods, df_hats))
        for aggr_by, df_hat in df_hats.items():
            print(f'- AGGREGATE BY {aggr_by.upper()}')
            for metric_name, metric_fn in metrics.items():
                error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
                print(f' {metric_name}: {error:.4f}')
    except Exception as e:
        print(f'[Skipping post-hoc analysis: {e}]')
 
    # ── GUARDAR RESULTADOS ESTRUCTURADOS ──────────────────────────────────────
    # Si GRIN_OUTPUT_PATH está definido, guarda arrays para análisis posterior
    output_path = os.environ.get('GRIN_OUTPUT_PATH', '')
    print(f"DEBUG: GRIN_OUTPUT_PATH = {output_path}")
    if output_path:
        print("DEBUG: Convirtiendo tensores a numpy...")
        y_true_np = y_true.detach().cpu().numpy()
        mask_np   = mask.detach().cpu().numpy()
        print(f"DEBUG: y_true_np shape={y_true_np.shape}, mask_np shape={mask_np.shape}")
 
        # Obtener el índice temporal del test si el dataset lo tiene
        try:
            timestamps = dataset.df.index[dm.test_slice].astype(str).tolist()
            print(f"DEBUG: timestamps obtenidos, longitud {len(timestamps)}")
        except Exception:
            print(f"DEBUG: Error obteniendo timestamps: {e}")
            timestamps = []
        print(f"DEBUG: Guardando en {output_path}")
        # Guardar como .npz comprimido — eficiente para arrays grandes
        np.savez_compressed(
            output_path,
            y_hat      = y_hat,
            y_true     = y_true_np.reshape(y_true_np.shape[0], y_true_np.shape[1], -1),
            mask       = dm.torch_dataset.eval_mask[dm.test_slice].reshape(
                            mask_np.shape[0], mask_np.shape[1], -1),  # ← eval_mask, no mask
            timestamps = np.array(timestamps, dtype=str),
        )
        print("DEBUG: Guardado exitoso")
        print(f'[Resultados guardados en: {output_path}.npz]')
    else:
        print("DEBUG: GRIN_OUTPUT_PATH no definido, no se guarda resultado")
 
    return y_true, y_hat, mask
 


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)