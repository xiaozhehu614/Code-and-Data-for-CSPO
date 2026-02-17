import numpy as np
import pandas as pd
from pathlib import Path
import logging
import gc
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import psutil
import sys
import contextlib

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling as RandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation as Mutation
from pymoo.termination import get_termination

# Global variables for worker processes
global_signal_matrix = None
global_matrix_shape = None

class OptimizationOutputCapture:
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.output_folder / "detailed_optimization_log.txt"
        
    @contextlib.contextmanager
    def capture_output(self):
        class TeeOutput:
            def __init__(self, file_path):
                self.file_path = Path(file_path)
                self.terminal = sys.stdout
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    f.write("")
                
            def write(self, message):
                self.terminal.write(message)
                self.terminal.flush()
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(message)
            
            def flush(self):
                self.terminal.flush()
        
        original_stdout = sys.stdout
        tee_output = TeeOutput(self.log_file_path)
        
        try:
            sys.stdout = tee_output
            yield
        finally:
            sys.stdout = original_stdout

def save_configuration_log(output_folder, config_info):
    config_file_path = Path(output_folder) / "optimization_configuration.txt"
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(config_info)

class SigmoidOptimizationConfig:
    def __init__(self):
        self.total_cores = cpu_count()
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        self.n_jobs_override = None
        self.parallel_threshold_override = None
        
        self.n_jobs_multiplier = 0.7
        self.parallel_threshold_multiplier = 3.0
        self.chunk_size = 10000
        
        self.enable_pool_restart = True
        self.pool_restart_threshold = 1000
        self.memory_limit_mb = 2000
        self.enable_fallback = True
        
        self.verbose_performance = True
        self.log_batch_sizes = False
        
    def get_optimal_n_jobs(self, signal_matrix):
        if self.n_jobs_override is not None:
            return self.n_jobs_override
        
        matrix_size = signal_matrix.shape[0] * signal_matrix.shape[1]
        matrix_memory_mb = matrix_size * 8 / (1024**2)
        
        base_n_jobs = max(1, int(self.total_cores * self.n_jobs_multiplier))
        
        if matrix_size > 5000000:
            n_jobs = min(base_n_jobs, 12)
        elif matrix_size > 2000000:
            n_jobs = min(base_n_jobs, 14)
        elif matrix_size > 1000000:
            n_jobs = min(base_n_jobs, 16)
        else:
            n_jobs = min(base_n_jobs, 18)
        
        if matrix_memory_mb > 500:
            n_jobs = min(n_jobs, 10)
        
        return max(2, min(n_jobs, self.total_cores - 2))
    
    def get_parallel_threshold(self, n_jobs):
        if self.parallel_threshold_override is not None:
            return self.parallel_threshold_override
        
        return max(n_jobs, int(n_jobs * self.parallel_threshold_multiplier))
    
    def print_configuration(self, signal_matrix):
        n_jobs = self.get_optimal_n_jobs(signal_matrix)
        parallel_threshold = self.get_parallel_threshold(n_jobs)
        
        print(f"\n{'='*60}")
        print(f"SIGMOID OPTIMIZATION CONFIGURATION")
        print(f"{'='*60}")
        print(f"System Info:")
        print(f"  Total CPU cores: {self.total_cores}")
        print(f"  Available memory: {self.available_memory_gb:.1f} GB")
        print(f"  Matrix shape: {signal_matrix.shape}")
        print(f"  Matrix size: {signal_matrix.shape[0] * signal_matrix.shape[1]:,} elements")
        print(f"  Estimated memory: {signal_matrix.shape[0] * signal_matrix.shape[1] * 8 / (1024**2):.1f} MB")
        print(f"\nProcess Pool Configuration:")
        print(f"  n_jobs: {n_jobs} ({'override' if self.n_jobs_override else 'auto'})")
        print(f"  parallel_threshold: {parallel_threshold} ({'override' if self.parallel_threshold_override else 'auto'})")
        print(f"  n_jobs_multiplier: {self.n_jobs_multiplier}")
        print(f"  parallel_threshold_multiplier: {self.parallel_threshold_multiplier}")
        print(f"\nAdvanced Settings:")
        print(f"  chunk_size: {self.chunk_size}")
        print(f"  memory_limit_mb: {self.memory_limit_mb}")
        print(f"  pool_restart: {self.enable_pool_restart} (every {self.pool_restart_threshold} evaluations)")
        print(f"  fallback_enabled: {self.enable_fallback}")
        print(f"{'='*60}\n")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nsga_sigmoid_optimization.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def init_worker(signal_matrix_flat, matrix_shape):
    global global_signal_matrix, global_matrix_shape
    global_signal_matrix = signal_matrix_flat.reshape(matrix_shape)
    global_matrix_shape = matrix_shape

def calculate_sigmoid_detection_score_worker(sensors_flat, chunk_size=10000):
    global global_signal_matrix
    
    sensors = sensors_flat.astype(np.bool_)
    
    if not sensors.any():
        return 0.0
    
    n_rows = global_signal_matrix.shape[0]
    
    if n_rows <= chunk_size:
        # Small matrix - process directly
        selected_signals = global_signal_matrix[:, sensors]
        max_signal_per_leak = np.max(selected_signals, axis=1)
        return np.mean(max_signal_per_leak)
    else:
        # Large matrix - process in chunks to manage memory
        total_sum = 0.0
        for i in range(0, n_rows, chunk_size):
            end = min(i + chunk_size, n_rows)
            chunk = global_signal_matrix[i:end, :]
            selected_signals = chunk[:, sensors]
            max_signal_per_leak = np.max(selected_signals, axis=1)
            total_sum += np.sum(max_signal_per_leak)
        
        return total_sum / n_rows

def evaluate_sigmoid_solution_worker(x):
    sensors = x.astype(np.bool_)
    n_sensors = np.sum(sensors).astype(np.int32)
    
    detection_score = calculate_sigmoid_detection_score_worker(sensors)
    
    f = np.array([-detection_score, n_sensors])
    g = np.array([1 - n_sensors, n_sensors - 50])
    
    return f, g

class ConfigurableParallelSigmoidSensorPlacementProblem(Problem):
    def __init__(self, signal_matrix, config=None, **kwargs):
        self.signal_matrix = signal_matrix
        self.config = config or SigmoidOptimizationConfig()
        
        self.n_jobs = self.config.get_optimal_n_jobs(signal_matrix)
        self.parallel_threshold = self.config.get_parallel_threshold(self.n_jobs)
        
        n_sensors = signal_matrix.shape[1]
        
        super().__init__(
            n_var=n_sensors,
            n_obj=2,
            n_constr=2,
            xl=0,
            xu=1,
            type_var=np.bool_,
            **kwargs
        )
        
        self.signal_matrix_flat = self.signal_matrix.flatten()
        self.matrix_shape = self.signal_matrix.shape
        
        self.evaluation_count = 0
        self.serial_count = 0
        self.parallel_count = 0
        self.total_eval_time = 0.0
        
        self._create_pool()
        
        logger.info(f"Configurable parallel sigmoid problem initialized:")
        logger.info(f"  Sensors: {n_sensors}")
        logger.info(f"  n_jobs: {self.n_jobs}")
        logger.info(f"  parallel_threshold: {self.parallel_threshold}")
        logger.info(f"  chunk_size: {self.config.chunk_size}")

    def _create_pool(self):
        if hasattr(self, 'pool'):
            try:
                self.pool.close()
                self.pool.join()
            except:
                pass
        
        self.pool = Pool(
            self.n_jobs, 
            initializer=init_worker, 
            initargs=(self.signal_matrix_flat, self.matrix_shape)
        )
        
        if self.config.verbose_performance:
            logger.info(f"Process pool created/recreated with {self.n_jobs} workers")

    def _calculate_detection_score(self, sensors):
        if not sensors.any():
            return 0.0
            
        chunk_size = self.config.chunk_size
        n_rows = self.signal_matrix.shape[0]
        
        if n_rows <= chunk_size:
            selected_signals = self.signal_matrix[:, sensors]
            max_signal_per_leak = np.max(selected_signals, axis=1)
            return np.mean(max_signal_per_leak)
        else:
            total_sum = 0.0
            for i in range(0, n_rows, chunk_size):
                end = min(i + chunk_size, n_rows)
                chunk = self.signal_matrix[i:end, :]
                selected_signals = chunk[:, sensors]
                max_signal_per_leak = np.max(selected_signals, axis=1)
                total_sum += np.sum(max_signal_per_leak)
            
            return total_sum / n_rows

    def _evaluate_serial(self, X):
        batch_size = X.shape[0]
        F = np.zeros((batch_size, 2))
        G = np.zeros((batch_size, 2))
        
        for i in range(batch_size):
            sensors = X[i].astype(np.bool_)
            n_sensors = np.sum(sensors).astype(np.int32)
            
            detection_score = self._calculate_detection_score(sensors)
            
            F[i] = [-detection_score, n_sensors]
            G[i] = [1 - n_sensors, n_sensors - 50]
        
        return F, G

    def _evaluate_parallel(self, X):
        try:
            results = self.pool.map(evaluate_sigmoid_solution_worker, X)
            F = np.array([r[0] for r in results])
            G = np.array([r[1] for r in results])
            return F, G
        except Exception as e:
            if self.config.enable_fallback:
                logger.warning(f"Parallel evaluation failed: {e}, falling back to serial")
                return self._evaluate_serial(X)
            else:
                raise

    def _evaluate(self, X, out, *args, **kwargs):
        eval_start_time = time.time()
        batch_size = X.shape[0]
        
        if self.config.log_batch_sizes:
            logger.debug(f"Evaluating batch of size {batch_size}")
        
        estimated_memory_mb = (batch_size * self.signal_matrix.shape[0] * 8) / (1024**2) 
        
        use_parallel = (
            batch_size >= self.parallel_threshold and 
            estimated_memory_mb <= self.config.memory_limit_mb
        )
        
        if use_parallel:
            F, G = self._evaluate_parallel(X)
            self.parallel_count += 1
        else:
            F, G = self._evaluate_serial(X)
            self.serial_count += 1
        
        eval_time = time.time() - eval_start_time
        self.total_eval_time += eval_time
        self.evaluation_count += 1
        
        if (self.config.enable_pool_restart and 
            self.evaluation_count % self.config.pool_restart_threshold == 0):
            self._create_pool()
            if self.config.verbose_performance:
                logger.info(f"Pool restarted after {self.evaluation_count} evaluations")
        
        if (self.config.verbose_performance and 
            self.evaluation_count % 100 == 0):
            avg_time = self.total_eval_time / self.evaluation_count
            logger.info(f"Evaluation stats: {self.evaluation_count} total, "
                       f"{self.parallel_count} parallel, {self.serial_count} serial, "
                       f"avg_time: {avg_time:.3f}s")
        
        out["F"] = F
        out["G"] = G
    
    def get_performance_stats(self):
        return {
            'total_evaluations': self.evaluation_count,
            'parallel_evaluations': self.parallel_count,
            'serial_evaluations': self.serial_count,
            'parallel_ratio': self.parallel_count / max(1, self.evaluation_count),
            'avg_eval_time': self.total_eval_time / max(1, self.evaluation_count)
        }
    
    def __del__(self):
        if hasattr(self, 'pool'):
            try:
                self.pool.close()
                self.pool.join()
            except:
                pass

class SerialSigmoidSensorPlacementProblem(Problem):
    
    def __init__(self, signal_matrix, **kwargs):
        self.signal_matrix = signal_matrix
        n_sensors = signal_matrix.shape[1]
        
        super().__init__(
            n_var=n_sensors,
            n_obj=2,
            n_constr=2,
            xl=0,
            xu=1,
            type_var=np.bool_,
            **kwargs
        )
        
        logger.info(f"Serial sigmoid problem initialized with {n_sensors} sensors")

    def _calculate_detection_score(self, sensors):
        if not sensors.any():
            return 0.0
            
        chunk_size = 10000
        n_rows = self.signal_matrix.shape[0]
        
        if n_rows <= chunk_size:
            selected_signals = self.signal_matrix[:, sensors]
            max_signal_per_leak = np.max(selected_signals, axis=1)
            return np.mean(max_signal_per_leak)
        else:
            total_sum = 0.0
            for i in range(0, n_rows, chunk_size):
                end = min(i + chunk_size, n_rows)
                chunk = self.signal_matrix[i:end, :]
                selected_signals = chunk[:, sensors]
                max_signal_per_leak = np.max(selected_signals, axis=1)
                total_sum += np.sum(max_signal_per_leak)
            
            return total_sum / n_rows

    def _evaluate(self, X, out, *args, **kwargs):
        batch_size = X.shape[0]
        F = np.zeros((batch_size, 2))
        G = np.zeros((batch_size, 2))
        
        for i in range(batch_size):
            sensors = X[i].astype(np.bool_)
            n_sensors = np.sum(sensors).astype(np.int32)
            
            detection_score = self._calculate_detection_score(sensors)
            
            F[i] = [-detection_score, n_sensors]
            G[i] = [1 - n_sensors, n_sensors - 50]
        
        out["F"] = F
        out["G"] = G

def calculate_avg_min_distance(selected_sensors, distance_matrix):
    if len(selected_sensors) < 2:
        return 0.0
    
    try:
        selected_distances = distance_matrix[np.ix_(selected_sensors, selected_sensors)].astype(float)
        
        np.fill_diagonal(selected_distances, np.inf)
        
        min_distances = np.min(selected_distances, axis=1)
        
        valid_distances = min_distances[np.isfinite(min_distances)]
        return np.mean(valid_distances) if len(valid_distances) > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Error in distance calculation: {e}")
        return -1.0

def process_sigmoid_solution(args):
    i, solution, detection_score, sensor_count, node_names, distance_matrix, signal_matrix = args
    
    selected_sensors = np.where(solution == 1)[0]
    sensor_node_ids = [node_names[j] for j in selected_sensors]
    
    try:
        avg_min_distance = calculate_avg_min_distance(selected_sensors, distance_matrix)
    except Exception as e:
        logger.warning(f"Error calculating distance for solution {i+1}: {e}")
        avg_min_distance = -1.0
    
    return {
        "Solution": i + 1,
        "CDS": detection_score,
        "Number_of_Sensors": sensor_count,
        "Average_Minimum_Distance": avg_min_distance,
        "Selected_Sensor_IDs": sensor_node_ids
    }

def load_sigmoid_data_files(input_parquet_file, distance_matrix_file):
    """Load and preprocess data files for sigmoid optimization"""
    logger.info("Loading sigmoid detection matrix data...")
    
    try:
        # Load sigmoid matrix
        signal_matrix_df = pd.read_parquet(input_parquet_file, engine='pyarrow')
        
        if 'Iteration' not in signal_matrix_df.columns:
            raise ValueError("Expected 'Iteration' column not found in sigmoid matrix")
        
        # Extract node names and convert to float numpy array
        node_names = signal_matrix_df.columns[1:].tolist()
        signal_matrix = signal_matrix_df.iloc[:, 1:].values.astype(float)
        
        logger.info(f"Loaded sigmoid matrix: {len(node_names)} nodes, {len(signal_matrix)} events")
        
    except Exception as e:
        logger.error(f"Error loading sigmoid matrix: {e}")
        raise
    
    logger.info("Loading distance matrix data...")
    
    try:
        # Load distance matrix
        distance_df = pd.read_parquet(distance_matrix_file, engine='pyarrow')
        
        # Handle index correctly
        if isinstance(distance_df.index, pd.RangeIndex):
            if len(distance_df.columns) > len(node_names) and distance_df.columns[0] in node_names:
                index_col = distance_df.columns[0]
                distance_df = distance_df.set_index(index_col)
        
        # Extract distance matrix with proper node ordering
        distance_matrix = distance_df.loc[node_names, node_names].values.astype(float)
        
        logger.info(f"Distance matrix shape: {distance_matrix.shape}")
        
        # Clean up memory
        del signal_matrix_df
        gc.collect()
        
        return signal_matrix, distance_matrix, node_names
        
    except Exception as e:
        logger.error(f"Error loading distance matrix: {e}")
        raise

def run_sigmoid_nsga_optimization(signal_matrix, config=None, parallel_method='auto', pop_size=1000, n_gen=500):
    if config is None:
        config = SigmoidOptimizationConfig()
    
    if parallel_method == 'auto':
        matrix_size = signal_matrix.shape[0] * signal_matrix.shape[1]
        cpu_cores = cpu_count()
        
        if matrix_size > 500000 and cpu_cores >= 4:
            parallel_method = 'parallel'
            logger.info("Auto-selected parallel processing")
        else:
            parallel_method = 'serial'
            logger.info("Auto-selected serial processing")
    
    if parallel_method == 'parallel':
        problem = ConfigurableParallelSigmoidSensorPlacementProblem(signal_matrix, config)
    else:
        problem = SerialSigmoidSensorPlacementProblem(signal_matrix)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=RandomSampling(),
        crossover=HUX(prob=0.9),
        mutation=Mutation(prob=1.0/signal_matrix.shape[1]),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_gen)
    
    logger.info(f"Starting optimization: method={parallel_method}, pop_size={pop_size}, n_gen={n_gen}")
    opt_start_time = time.time()
    
    try:
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            verbose=True,
            save_history=False
        )
    except Exception as e:
        logger.error(f"Sigmoid optimization failed: {e}")
        raise
    finally:
        if hasattr(problem, 'get_performance_stats'):
            stats = problem.get_performance_stats()
            logger.info(f"Performance statistics: {stats}")
        
        if hasattr(problem, 'pool'):
            try:
                problem.pool.close()
                problem.pool.join()
            except:
                pass
    
    opt_time = time.time() - opt_start_time
    logger.info("Optimization complete")
    
    return res, opt_time

def save_sigmoid_results(results_df, output_folder):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_csv_path = output_path / "pareto_optimal_solutions_summary.csv"
    output_parquet_path = output_path / "pareto_optimal_solutions.parquet"
    
    results_df.to_csv(output_csv_path, index=False)
    
    results_df_for_parquet = results_df.copy()
    results_df_for_parquet["Selected_Sensor_IDs"] = results_df_for_parquet["Selected_Sensor_IDs"].apply(
        lambda x: ';'.join(map(str, x)) if isinstance(x, list) else str(x)
    )
    results_df_for_parquet.to_parquet(output_parquet_path, engine='pyarrow', index=False)
    
    logger.info(f'Sigmoid results saved to {output_csv_path} and {output_parquet_path}')
    return output_csv_path, output_parquet_path

def main():
    global logger
    logger = setup_logging()
    start_time = time.time()
    
    input_parquet_file = 'output/Sigmoid_Matrix/Sigmoid_Detection_Matrix_Processed.parquet'
    distance_matrix_file = 'output/Distance_Matrix/Distance_Matrix_Processed.parquet'
    output_folder = 'output/CSPO'
    
    parallel_method = 'parallel'
    pop_size = 1000
    n_gen = 500
    
    config = SigmoidOptimizationConfig()
    
    config.n_jobs_multiplier = 0.7
    config.parallel_threshold_multiplier = 2.5
    config.chunk_size = 15000
    config.memory_limit_mb = 100000
    config.pool_restart_threshold = 500
    config.enable_pool_restart = True
    config.enable_fallback = True
    config.verbose_performance = False
    config.log_batch_sizes = False

    config_info = f"""NSGA-II Sigmoid Sensor Placement (CSPO)
===========================
Input Sigmoid Matrix: {input_parquet_file}
Input Distance Matrix: {distance_matrix_file}
Output Folder: {output_folder}

Algorithm:
    pop_size: {pop_size}
    n_gen: {n_gen}
    parallel_method: {parallel_method}

Parallel:
    n_jobs_multiplier: {config.n_jobs_multiplier}
    parallel_threshold_multiplier: {config.parallel_threshold_multiplier}
    chunk_size: {config.chunk_size}
    memory_limit_mb: {config.memory_limit_mb}
    pool_restart_threshold: {config.pool_restart_threshold}
    enable_pool_restart: {config.enable_pool_restart}
    enable_fallback: {config.enable_fallback}
    verbose_performance: {config.verbose_performance}
"""
    
    output_capture = OptimizationOutputCapture(output_folder)
    
    save_configuration_log(output_folder, config_info)
    
    logger.info(f"Configuration: parallel_method={parallel_method}, pop_size={pop_size}, n_gen={n_gen}")
    
    try:
        signal_matrix, distance_matrix, node_names = load_sigmoid_data_files(
            input_parquet_file, distance_matrix_file
        )
        
        with output_capture.capture_output():
            res, opt_time = run_sigmoid_nsga_optimization(
                signal_matrix, 
                config=config,
                parallel_method=parallel_method, 
                pop_size=pop_size, 
                n_gen=n_gen
            )
        
        logger.info("Processing sigmoid optimization results...")
        pareto_solutions = res.X
        pareto_front = res.F
        
        detection_scores = -pareto_front[:, 0]
        sensor_counts = pareto_front[:, 1].astype(np.int32)
        
        sorted_indices = np.argsort(-detection_scores)
        detection_scores = detection_scores[sorted_indices]
        sensor_counts = sensor_counts[sorted_indices]
        pareto_solutions = pareto_solutions[sorted_indices]
        
        logger.info("Processing Pareto optimal solutions in parallel...")
        num_cores = max(1, min(cpu_count() - 1, 8))
        
        process_args = [
            (i, solution, detection_scores[i], sensor_counts[i], node_names, distance_matrix, signal_matrix)
            for i, solution in enumerate(pareto_solutions)
        ]
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(process_sigmoid_solution, process_args))
        
        results_df = pd.DataFrame(results)
        
        save_sigmoid_results(results_df, output_folder)
        
        total_time = time.time() - start_time
        post_process_time = total_time - opt_time
        
        logger.info(f"Solutions: {len(results)}")
        logger.info(
            f"Score: avg={np.mean(detection_scores):.4f}, max={np.max(detection_scores):.4f}"
        )
        logger.info(
            f"Sensors: range=[{np.min(sensor_counts)}-{np.max(sensor_counts)}]"
        )
        logger.info(f"Outputs: {output_folder}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    mp.freeze_support()
    main()