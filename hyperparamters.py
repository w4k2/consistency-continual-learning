from main import *
import mlflow
import distutils.util


def main():
    args = parse_args()
    args.nested_run = True
    args.debug = False
    args.train_on_experiences = 1  # TODO change to 10
    args.seed = 3141592
    if args.run_name is None:
        args.run_name = f'{args.method} hyperparameters'
    run_name = args.run_name

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        id = mlflow.create_experiment(args.experiment)
        experiment = client.get_experiment(id)
    experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        active_run = mlflow.active_run()
        parrent_run_id = active_run.info.run_id
        grid_search(args, experiment_id)

    n_repeats = 5
    args = select_best_paramters(args, client, experiment, parrent_run_id)
    args.train_on_experiences = args.n_experiences
    args.forgetting_stopping_threshold = 1.0
    args.seed = 0

    with mlflow.start_run(experiment_id=experiment_id, run_name=f'{run_name} final'):
        for repeat in range(n_repeats):
            args.run_name = f'{args.method} final run {repeat}'
            args.seed += 1
            run_experiment(args)


def grid_search(args, experiment_id, num_repeats=2):  # TODO change num repeates to 3
    for alpha in (0.1, 0.3):  # (0.1, 0.3, 0.5, 0.7, 0.9): # TODO change
        run_name = f'{args.method}, alpha={alpha}'
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
            for i in range(num_repeats):
                print(f'{args.method}, alpha = {alpha}')
                args.run_name = f'{args.method}, alpha={alpha}, repeat={i}'
                args.alpha = alpha
                args.seed += i

                run_experiment(args)


def select_best_paramters(args, client, experiment, parrent_run_id):
    experiment_id = experiment.experiment_id
    best_run = select_best(client, parrent_run_id, experiment_id, method='avrg_acc')
    best_parameters = best_run.data.params

    arg_names = list(vars(args).keys())
    for name in arg_names:
        value = best_parameters[name]
        arg_type = type(getattr(args, name))
        if arg_type == int:
            value = int(value)
        elif arg_type == float:
            value = float(value)
        elif arg_type == bool:
            value = distutils.util.strtobool(value)
        else:
            value = arg_type(value)
        setattr(args, name, value)

    print('\nbest args')
    for name, value in vars(args).items():
        print(f'\t{name}: {value}, type = {type(value)}')

    return args


def select_best(client, parrent_run_id, experiment_id, method='avrg_acc'):
    selected_runs = select_runs_with_parent(client, parrent_run_id, experiment_id)
    selected_runs = [select_runs_with_parent(client, run.info.run_id, experiment_id) for run in selected_runs]

    best_run = None
    best = 0.0
    for run_reapeats in selected_runs:
        metric_avrg = 0.0
        for run in run_reapeats:
            metric_avrg += get_metric(run, method)
        metric_avrg /= len(run_reapeats)
        if metric_avrg > best:
            best = metric_avrg
            best_run = run

    return best_run


def select_runs_with_parent(client, parrent_run_id, experiemnt_id):
    runs_df = mlflow.search_runs(experiment_ids=[experiemnt_id])

    selected_runs = []
    for _, row in runs_df.iterrows():
        run_id = row['run_id']
        run = client.get_run(run_id)
        run_data = run.data
        if 'mlflow.parentRunId' not in run_data.tags:
            continue
        parent_run = run_data.tags['mlflow.parentRunId']
        if parent_run != parrent_run_id:
            continue
        selected_runs.append(run)

    return selected_runs


def get_metric(run, method='avrg_acc'):
    run_metrics = run.data.metrics
    if method == 'first_task':
        current = run_metrics['test_accuracy_task_0']
    elif method == 'avrg_acc':
        current = run_metrics['avrg_test_acc']
    else:
        raise ValueError('Invalid method argument value in select_best call')
    return current


if __name__ == '__main__':
    main()
