import argparse
import yaml
from pathlib import Path

# Absolute imports from project structure
from Preprocessing.load_data_pressure import load_data_pressure_from_config
from Trials import TRIALS
import torch

def load_yaml_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main(config_path):
    import os
    print("[DEBUG] Entered main(), config_path =", config_path, flush=True)
    # Step 1: Load config
    config = load_yaml_config(config_path)
    print("[DEBUG] Loaded config OK", flush=True)
    # Step 1.1: Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] Device is", device, flush=True)
    # Step 2: Resolve train/test cases from Trials.py
    trial_name = config["trial_name"]
    if trial_name not in TRIALS:
        raise ValueError(f"[ERROR] Trial '{trial_name}' not found. Available: {list(TRIALS.keys())}")
    train_cases = TRIALS[trial_name]["train"]
    test_cases = TRIALS[trial_name]["test"]
    print(f"[INFO] Trial '{trial_name}': train={train_cases}, test={test_cases}")

    # Step 3: Create output directory and save config
    base_output = Path(config["paths"]["output_dir"])  # e.g. outputs/run2/single_phill_minmax/w4m4d3
    run_name = config["paths"]["name"]  # e.g., "kan_single_phill_minmax_w5m5d8_lr0.01"
    output_dir = base_output / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_used.yaml", 'w') as f:
        yaml.dump(config, f)
    print(f"[INFO] Output will be saved to: {output_dir}")

    # Redirect internal MultKAN saves to a subfolder inside this run dir
    os.environ['KAN_MODEL_DIR'] = str(output_dir / "model")

    # Step 4: Load tensors from config and normalize
    print("[DEBUG] About to load data", flush=True)
    data_path = os.path.join(os.getcwd(), 'Data')
    data = load_data_pressure_from_config(config, train_cases, test_cases, data_path)
    print("[DEBUG] Data loaded; X_train cases:", [x.shape for x in data["X_train"]], flush=True)

    #import normalization functions
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train = data["X_train"]
    X_test = data["X_test"]

    # ───────── LABEL NORMALISATION ──────────────────────────────────
    norm_labels = config['training'].get('norm_labels', False)
    if norm_labels:
        from sklearn.preprocessing import MinMaxScaler  # or StandardScaler
        label_scaler = MinMaxScaler()  # fit on train set only

        # helper to transform list-of-tensors <--> numpy
        def _scale_list(t_list, fn):
            return [torch.tensor(fn(t.cpu().numpy()).astype('float32'),
                            device=device)for t in t_list ]

        y_train = _scale_list(y_train, label_scaler.fit_transform)
        y_test = _scale_list(y_test, label_scaler.transform)

        # save the scaler for later inverse-transform / deployment
        import joblib, os
        scaler_path = output_dir / "label_scaler.pkl"
        joblib.dump(label_scaler, scaler_path)
        print(f"[INFO] Label scaler saved to {scaler_path}", flush=True)
    # ─────────────────────────────────────────────────────────────────

    from features.do_full_norms import apply_two_step_normalization, normalize_x_test

    # === Normalize input features using updated two-step method ===
    X_train_normed, x_scalers = apply_two_step_normalization(
        X_train,
        config=config,
        input_feature_names=config["features"]["input"],
        case_names=train_cases,
        grid_dicts=data["grid_dicts_train"],
        device=device
    )

    X_test_normed, _, _ = normalize_x_test(
        X_test,
        config=config,
        input_feature_names=config["features"]["input"],
        case_names=test_cases,
        grid_dicts=data["grid_dicts_test"],
        device=device,
        scalers=x_scalers
    )

    print(f'[DEBUG] X_train_normed shape: {X_train_normed[0].shape}', flush=True)
    print(f'[DEBUG] y_train_normed shape: {y_train[0].shape}', flush=True)

    # Add in transport term if config says so
    if config['features']['transport_feature'] == True:
        from features.add_transport import add_transport_feature
        print('[INFO] Adding Transport Feature')
        X_train_normed = add_transport_feature(
            x_tensor_list=X_train_normed,
            case_names=train_cases,
            config=config,
            grid_dicts=data["grid_dicts_train"],
            data_path=data_path,
            device=device
        )
        X_test_normed = add_transport_feature(
            x_tensor_list=X_test_normed,
            case_names=test_cases,
            config=config,
            grid_dicts=data["grid_dicts_test"],
            data_path=data_path,
            device=device
        )

    # make KDE plots
    from Plotting.plot_input_kde import plot_input_kde
    kde_dir = output_dir / "kde"
    kde_dir.mkdir(parents=True, exist_ok=True)
    plot_input_kde(
        x_train_normed=X_train_normed,
        x_test_normed=X_test_normed,
        train_cases=train_cases,
        test_cases=test_cases,
        config=config,
        save_dir=kde_dir,
    )

    # Optional: for GNNs
    if "edge_index_train" in data:
        edge_train = data["edge_index_train"]
        edge_test = data["edge_index_test"]

    print(f"[DEBUG] Data feature normed, norm used:{config['training']['feature_norm']} " , flush=True)
    print(f"[DEBUG] Data scale normed, norm used:{config['training']['scale_norm']} " , flush=True)

    # Step 5: Select and run model based on config
    model_type = config["model"]["type"].lower()

    # Preinitialize training outputs
    predictions = None
    loss_df = None
    model = None
    best_epoch = None
    model_state_dict = None
    optimizer = None
    print("[INFO] Training model type {model_type}", flush=True)
    if model_type == "fcn":
        from Models.FCN import runSimple_model
        print('[INFO] FCN sim starting.', flush=True)
        y_pred, loss_df, best_model, best_epoch, best_state_dict, optimizer = runSimple_model(
            X_train_normed, y_train, X_test_normed,
            config=config,
            directory=output_dir,
            device=device
        )
        print('[INFO] FCN sim ended.', flush=True)

    elif model_type in ['bnn', 'branch', 'bnn']:
        from Models.FCN2 import runBranch_model
        print('[INFO] BNN sim starting.', flush=True)

        y_pred, loss_df, best_model, best_epoch, best_state_dict, optimizer = runBranch_model(
            X_train_normed, y_train, X_test_normed, config=config, directory=output_dir,device=device
        )
        print('[INFO] BNN sim ending.', flush=True)

    elif model_type == "kan":
        from Models.Kan import runKAN_model
        # Determine the true number of input features
        num_input_features = len(config['features']['input'])
        if config['features']['transport_feature'] == True:
            num_input_features += 1  # +1 for T_transport

        # Validate input/output dimensions
        expected_input = config['model']['shape'][0][0]
        expected_output = config['model']['shape'][-1][0]
        if num_input_features != expected_input:
            raise ValueError(
                f"[ERROR] Input feature count ({num_input_features}) does not match model.shape[0][0] ({expected_input})")
        if len(config['features']['output']) != expected_output:
            raise ValueError(
                f"[ERROR] Output feature count ({len(config['features']['output'])}) does not match model.shape[-1][0] ({expected_output})")

        print('[INFO] KAN sim starting.', flush=True)
        y_pred, loss_df, best_model, best_epoch, best_state_dict, optimizer = runKAN_model(
            X_train_normed, y_train, X_test_normed,
            config=config,
            output_dir=output_dir, device=device)
        print('[INFO] KAN sim starting.', flush=True)

    # TODO: add support for additional model types (e.g., GNNs, old_FCN) here
    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported yet.")

    print('[INFO] Post processing: saving started ', flush=True)

    # ───────── INVERSE SCALE PRED & TEST LABELS (if needed) ─────────
    if norm_labels:
        def _inverse_scale_list(t_list):
            return [
                torch.tensor(
                    label_scaler.inverse_transform(t.cpu().numpy()),
                    device='cpu'  # keep on CPU for plotting / metrics
                )
                for t in t_list
            ]

        y_pred = _inverse_scale_list(y_pred)
        y_test = _inverse_scale_list(y_test)
    # ─────────────────────────────────────────────────────────────────


    # Step 6: Save predictions and test loss info
    from utils.Get_test_losses import compute_test_losses

    # Create subdirectories
    pred_dir = output_dir / "predictions"
    loss_dir = output_dir / "test_loss_info"
    pred_dir.mkdir(exist_ok=True)
    loss_dir.mkdir(exist_ok=True)

    # TODO: When implementing GNNs, use a graph-aware test loss function

    # Compute test loss
    total_test_loss, avg_test_loss = compute_test_losses(
        best_model, X_test_normed, y_test, config)

    torch.save(y_pred, pred_dir / "y_pred_norm.pt")
    torch.save(y_test, pred_dir / "y_test_norm.pt")
    with open(pred_dir / "test_cases.txt", 'w') as f:
        f.writelines([f"{case}\n" for case in test_cases])
    print(f"[INFO] Saved predictions and test set to {pred_dir}", flush=True)

    # Save loss info
    with open(loss_dir / "loss_summary.txt", 'w') as f:
        f.write(f"Total test loss: {total_test_loss:.6f}\n")
        f.write(f"Average test loss: {avg_test_loss:.6f}\n")
    print(f"[INFO] Saved loss summary to {loss_dir}")

    print('[INFO] Post processing: accuracy metrics started ', flush=True)
    # === Step 7: Accuracy metrics and summary ===
    from utils.make_accuracies_pressure import make_abs_diff_metrics, save_metrics_summary

    # 7.1 Compute all metrics (includes abs‐diff, Pearson r, and net accuracy)
    metrics = make_abs_diff_metrics(y_pred, y_test, device=device)
    try:
        metrics['num_params'] = float(best_model.parameters())
    except:
        print('[ERROR] best_model.parameters() were found in the model.', flush=True)

    # 7.2 Write them out under a real “metrics” folder
    metrics_dir = output_dir / "metrics"
    save_metrics_summary(metrics, metrics_dir, model_name="model")

    print(f"[INFO] Saved accuracy metrics to {metrics_dir}/model_metrics.txt", flush=True)

    # Step 8: Plot predictions (truth vs prediction vs difference)
    plot_dir = output_dir / "predictions" / "plots"
    if len(config['features']['output']) > 1:
        from Plotting.Plot_all import plot_all_cases
        plot_all_cases(list(y_test), list(y_pred),case_names=test_cases, save_dir=plot_dir, grid_dicts=data["grid_dicts_test"], out_features=config['features']['output'])
    elif len(config['features']['output']) == 1:
        from Plotting.plot_pressure import plot_all_cases
        plot_all_cases(list(y_test), list(y_pred),case_names=test_cases, save_dir=plot_dir, grid_dicts=data["grid_dicts_test"])
    print(f"[INFO] Saved prediction plots to {plot_dir}")

    # === Step 9: Plot training loss (all cases) ===
    from Plotting.plot_losses import plot_all_case_losses

    loss_plot_dir = output_dir / "loss_plots"
    os.makedirs(loss_plot_dir, exist_ok=True)
    plot_all_case_losses(
        losses_df=loss_df,
        config=config,
        output_dir=loss_plot_dir)
    print(f"[INFO] Saved all training loss plots to: {loss_plot_dir}", flush=True)

    if config.get("evaluate_training_predictions", False):
        from Plotting.Plot_all import plot_all_cases
        from training_utils.Make_predictions import make_preds
        train_predictions = make_preds(best_model.to(device), [x.to(device) for x in X_train_normed])
        plot_all_cases(y_train, list(train_predictions), case_names= train_cases, save_dir =output_dir / "train_predictions" / "plots",
                       grid_dicts=data["grid_dicts_train"], out_features=config['features']['output'])
    print(f"[RAD INFO] WE FINISHED BROSKI", flush=True)

    # [Next steps go here...]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NN simulation from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)


