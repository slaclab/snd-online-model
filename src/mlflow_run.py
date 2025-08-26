"""
mlflow_run.py
-------------
Provides a context manager class for managing MLflow experiment runs with automatic naming and tracking URI setup.

Classes
-------
MLflowRun
    Context manager for MLflow runs, automatically sets up experiment and run naming.
"""
import logging
import mlflow

logger = logging.getLogger(__name__)

class MLflowRun:
    """
    Context manager for MLflow runs.

    Automatically sets the tracking URI and experiment name, and generates a unique run name
    based on previous runs. Ensures that MLflow runs are properly started and ended.

    Parameters
    ----------
    tracking_uri : str, optional
        The MLflow tracking server URI.
    experiment_name : str, optional
        The name of the MLflow experiment.
    run_prefix : str, optional
        Prefix for run names to distinguish runs.
    """
    def __init__(
        self,
        tracking_uri="https://ard-mlflow.slac.stanford.edu",
        experiment_name="snd-online-model",
        run_prefix="SND Online Model Run",
    ):
        self.run_prefix = run_prefix
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = self.setup_experiment()

    def __enter__(self):
        """
        Start an MLflow run and return the run object.
        """
        self.run = mlflow.start_run(run_name=self.run_name)
        logger.info(f"Started MLflow run: {self.run_name}")
        return self.run

    def __exit__(self, exc_type, exc_value, traceback):
        """
        End the MLflow run when exiting the context.
        """
        mlflow.end_run()

    def setup_experiment(self):
        """
        Set up the MLflow experiment and generate a unique run name.

        Returns
        -------
        str
            The generated run name for the new MLflow run.
        """
        logger.debug("Setting up MLflow experiment...")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)

        # Get next run name
        all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_numbers = []

        for run in all_runs:
            tag = run.data.tags.get("mlflow.runName", "")
            if tag.startswith(self.run_prefix):
                try:
                    num = int(tag.replace(self.run_prefix, "").strip())
                    run_numbers.append(num)
                except ValueError:
                    continue

        next_run_number = max(run_numbers, default=0) + 1
        return self.run_prefix + str(next_run_number)
