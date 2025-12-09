import argparse
import subprocess
import json
import os
import sys
import glob
from itertools import product
import pandas as pd
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test multiple alpha and beta hyperparameters for 3CD evaluation"
    )
    
    parser.add_argument("--model", type=str, required=True, help="Model type (e.g., llava-1.5)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--pope_type", type=str, default="random", help="POPE type")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--data_path", type=str, default="/home/donut2024/coco2014", help="Data path")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images")
    parser.add_argument("--max_new_tokens", type=int, default=4, help="Max new tokens")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./paper_result/", help="Output directory")
    parser.add_argument("--gt_seg_path", type=str, default="pope_coco/coco_ground_truth_segmentation.json", help="Ground truth segmentation path")
    
    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
        help="List of alpha values to test"
    )
    parser.add_argument(
        "--beta_values",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.75, 1.0],
        help="List of beta values to test"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip combinations that already have results"
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="hyperparameter_test_summary.csv",
        help="Output CSV file for summary results"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evaluations in parallel (requires multiple GPUs or sequential execution)"
    )
    
    return parser.parse_args()


def run_evaluation(args, alpha, beta):
    """Run a single evaluation with given alpha and beta values."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generation_script = os.path.join(script_dir, "generation_3CD_pope.py")
    
    if not os.path.exists(generation_script):
        generation_script = "generation_3CD_pope.py"
    
    cmd = [
        sys.executable,
        generation_script,
        "--model", args.model,
        "--model_path", args.model_path,
        "--pope_type", args.pope_type,
        "--gpu-id", str(args.gpu_id),
        "--data_path", args.data_path,
        "--num_samples", str(args.num_samples),
        "--num_images", str(args.num_images),
        "--max_new_tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--output_dir", args.output_dir,
        "--gt_seg_path", args.gt_seg_path,
        "--alpha", str(alpha),
        "--beta", str(beta),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running evaluation: alpha={alpha}, beta={beta}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Completed: alpha={alpha}, beta={beta}")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: alpha={alpha}, beta={beta}")
        print(f"Error: {e.stderr}")
        return False, e.stderr


def find_result_file(args, alpha, beta):
    """Find the result JSON file for a given alpha/beta combination."""
    base_dir = os.path.join(args.output_dir, "pope", args.model)
    if not os.path.exists(base_dir):
        return None
    
    alpha_str = str(alpha).rstrip('0').rstrip('.') if '.' in str(alpha) else str(alpha)
    beta_str = str(beta).rstrip('0').rstrip('.') if '.' in str(beta) else str(beta)
    
    pattern = f"{args.pope_type}_*_{args.model}_3CD_seed_{args.seed}_alpha_{alpha_str}_beta_{beta_str}_max_tokens_{args.max_new_tokens}_samples_{args.num_images}_results.json"
    matches = glob.glob(os.path.join(base_dir, pattern))
    
    if not matches:
        pattern = f"{args.pope_type}_*_{args.model}_3CD_seed_{args.seed}_alpha_{alpha}_beta_{beta}_max_tokens_{args.max_new_tokens}_samples_{args.num_images}_results.json"
        matches = glob.glob(os.path.join(base_dir, pattern))
    
    if not matches:
        pattern = f"{args.pope_type}_*_{args.model}_3CD_seed_{args.seed}_alpha_*_beta_*_max_tokens_{args.max_new_tokens}_samples_{args.num_images}_results.json"
        all_matches = glob.glob(os.path.join(base_dir, pattern))
        for match in all_matches:
            filename = os.path.basename(match)
            try:
                alpha_match = None
                beta_match = None
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part == 'alpha' and i + 1 < len(parts):
                        alpha_match = parts[i + 1]
                    if part == 'beta' and i + 1 < len(parts):
                        beta_match = parts[i + 1]
                
                if alpha_match and beta_match:
                    try:
                        file_alpha = float(alpha_match)
                        file_beta = float(beta_match)
                        if abs(file_alpha - alpha) < 0.001 and abs(file_beta - beta) < 0.001:
                            matches.append(match)
                    except ValueError:
                        continue
            except Exception:
                continue
    
    if matches:
        return max(matches, key=os.path.getmtime)
    return None


def load_result(result_path):
    """Load results from a JSON file."""
    try:
        with open(result_path, 'r') as f:
            content = f.read().strip()
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    lines = content.split('\n')
                    for line in reversed(lines):
                        if line.strip():
                            return json.loads(line)
    except Exception as e:
        print(f"Error loading {result_path}: {e}")
    return None


def collect_results(args, alpha_values, beta_values):
    """Collect all results from completed evaluations."""
    results = []
    
    for alpha, beta in product(alpha_values, beta_values):
        result_file = find_result_file(args, alpha, beta)
        if result_file:
            result_data = load_result(result_file)
            if result_data:
                result_data['alpha'] = alpha
                result_data['beta'] = beta
                result_data['result_file'] = result_file
                results.append(result_data)
            else:
                print(f"Warning: Could not load results for alpha={alpha}, beta={beta}")
        else:
            print(f"Warning: No result file found for alpha={alpha}, beta={beta}")
    
    return results


def create_summary_table(results, output_path):
    """Create a summary table from results."""
    if not results:
        print("No results to summarize.")
        return
    
    df = pd.DataFrame(results)
    
    cols = ['alpha', 'beta', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    if 'result_file' in df.columns:
        cols.append('result_file')
    
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]
    
    if 'F1 Score' in df.columns:
        df = df.sort_values('F1 Score', ascending=False)
    elif 'Accuracy' in df.columns:
        df = df.sort_values('Accuracy', ascending=False)
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Summary saved to: {output_path}")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TEST SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    if 'F1 Score' in df.columns:
        best = df.iloc[0]
        print(f"\nBest F1 Score: {best['F1 Score']:.4f}")
        print(f"  Alpha: {best['alpha']}, Beta: {best['beta']}")
        if 'Accuracy' in best:
            print(f"  Accuracy: {best['Accuracy']:.4f}")
        if 'Precision' in best:
            print(f"  Precision: {best['Precision']:.4f}")
        if 'Recall' in best:
            print(f"  Recall: {best['Recall']:.4f}")
    
    return df


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    combinations = list(product(args.alpha_values, args.beta_values))
    total = len(combinations)
    
    print(f"\n{'='*80}")
    print(f"3CD Hyperparameter Testing")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"POPE Type: {args.pope_type}")
    print(f"Alpha values: {args.alpha_values}")
    print(f"Beta values: {args.beta_values}")
    print(f"Total combinations: {total}")
    print(f"{'='*80}\n")
    
    to_run = []
    already_done = []
    
    for alpha, beta in combinations:
        if args.skip_existing:
            result_file = find_result_file(args, alpha, beta)
            if result_file and load_result(result_file):
                already_done.append((alpha, beta))
                print(f"⏭ Skipping (already exists): alpha={alpha}, beta={beta}")
                continue
        to_run.append((alpha, beta))
    
    if already_done:
        print(f"\nSkipped {len(already_done)} existing evaluations.")
    
    if not to_run:
        print("\nAll evaluations already completed. Collecting results...")
    else:
        print(f"\nRunning {len(to_run)} new evaluations...")
        start_time = time.time()
        
        for i, (alpha, beta) in enumerate(to_run, 1):
            print(f"\n[{i}/{len(to_run)}] ", end="")
            success, error = run_evaluation(args, alpha, beta)
            if not success:
                print(f"Failed evaluation for alpha={alpha}, beta={beta}")
                if error:
                    print(f"Error details: {error}")
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Completed {len(to_run)} evaluations in {elapsed:.2f} seconds")
        print(f"{'='*80}\n")
    
    print("Collecting results...")
    results = collect_results(args, args.alpha_values, args.beta_values)
    
    if results:
        summary_path = os.path.join(args.output_dir, args.output_summary)
        df = create_summary_table(results, summary_path)
        
        json_path = summary_path.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {json_path}")
    else:
        print("No results found. Please check that evaluations completed successfully.")
    
    print("\nHyperparameter testing complete!")


if __name__ == "__main__":
    main()

