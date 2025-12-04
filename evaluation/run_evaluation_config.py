"""
User configuration for evaluation framework.

Edit this file to specify which projection configurations and datasets to evaluate.
Then run: python evaluation/run_evaluation_config.py

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-04
"""

# Configuration JSON file to use
CONFIG_JSON = "evaluation/projection_configs/test_configs_v1.json"

# Datasets to evaluate on (extensible list - designed to support multiple datasets)
DATASETS = ["bomni"]  # Only BOMNI for now, will extend later

# Dataset root paths (must specify root for each dataset in DATASETS list)
DATASET_ROOTS = {
    ""
    "bomni": "datasets/all-datasets/BOMNI-production"
    # Add other dataset roots when ready
}

# Verbosity
VERBOSE = True

# Maximum images to process per dataset (None = all images)
# Use a small number for testing, None for full evaluation
MAX_IMAGES = None  # Set to 5 for quick testing

if __name__ == "__main__":
    from evaluation.run_projection_evaluation import ProjectionEvaluator

    print("="*80)
    print("Projection Configuration Evaluation")
    print("="*80)
    print(f"\nConfiguration file: {CONFIG_JSON}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Verbose: {VERBOSE}")
    print(f"Max images per dataset: {MAX_IMAGES if MAX_IMAGES else 'All'}")
    print("="*80)
    print()

    # Create evaluator
    evaluator = ProjectionEvaluator(
        config_json_path=CONFIG_JSON,
        dataset_roots=DATASET_ROOTS,
        verbose=VERBOSE,
        max_images=MAX_IMAGES
    )

    # Run evaluation
    evaluator.run_evaluation(dataset_names=DATASETS)

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
