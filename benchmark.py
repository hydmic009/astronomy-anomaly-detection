import os
import glob
import pandas as pd
import torch  # Uncomment if you need to use PyTorch directly
import numpy as np
import time
from PIL import Image
import google.generativeai as genai
import matplotlib.pyplot as plt

# ASTRONOMALY imports
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning

# --- Configuration ---
# Number of top anomalies to send to Gemini for active learning
N_FOR_ACTIVE_LEARNING = 20
# For the zero-shot benchmark, you might not want to run all ~60k images.
# Set to a number (e.g., 50) for a sample, or None to run on all.
N_FOR_GEMINI_ZERO_SHOT = 20

# --- Gemini API Helper Functions ---

def setup_gemini():
    """Configures the Gemini API with the key from environment variables."""
    # Ensure you have your GOOGLE_API_KEY set as an environment variable
    # For example, in your terminal: export GOOGLE_API_KEY="your_api_key_here"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    print("Gemini API configured.")

def get_gemini_response(image_path, prompt, model_name="gemini-2.0-flash-lite"):
    """
    Sends a single image and a prompt to the Gemini API and gets a response.
    """
    try:
        model = genai.GenerativeModel(model_name)
        img = Image.open(image_path)
        response = model.generate_content([prompt, img], stream=False)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"An error occurred with Gemini API for {os.path.basename(image_path)}: {e}")
        # Wait a bit in case of rate limiting or other transient errors
        time.sleep(5)
        return ""

def parse_score_from_response(text, max_score=5):
    """
    Parses a numerical score from Gemini's text response. Robust to extra text.
    """
    # Find all numbers in the string
    numbers = [int(s) for s in text.split() if s.isdigit()]
    if numbers:
        # Return the first number found, clipped to the max score
        return min(numbers[0], max_score)
    return -1  # Return -1 to indicate failure

# --- Evaluation Helper Functions ---

def rank_weighted_score(ranked_indices, true_anomaly_indices):
    """
    Calculates the Rank Weighted Score (RWS) as defined in the ASTRONOMALY paper.
    """
    N_true = len(true_anomaly_indices)
    N_scored = len(ranked_indices)
    N = min(N_true, N_scored)

    if N == 0:
        return 0.0

    indicator = pd.Series(
        ranked_indices.isin(true_anomaly_indices).astype(int),
        index=ranked_indices
    )
    weights = np.arange(N, 0, -1)
    vals = indicator.iloc[:N].values
    score = np.sum(weights * vals)
    s0 = N * (N + 1) / 2
    return score / s0 if s0 > 0 else 0.0

def evaluate_performance(scores_df, ground_truth_indices, top_k=100):
    """
    Evaluates the performance of a ranking, correctly handling path differences.
    """
    if scores_df.empty:
        return {'recall_at_k': 0, 'rws': 0, 'top_k': top_k, 'found_in_top_k': 0}

    sorted_df = scores_df.sort_values('score', ascending=False)

    # Get the original indices from astronomaly (which include paths)
    ranked_indices_with_path = sorted_df.index

    # Create a new series of just the basename IDs for comparison with ground truth
    ranked_ids_only = pd.Index(ranked_indices_with_path.map(
        lambda x: os.path.splitext(os.path.basename(x))[0]
        ))
    top_k_ranked = ranked_ids_only[:top_k]

    if len(ground_truth_indices) == 0:
        recall_at_k = 0.0
        true_positives_in_top_k = 0
    else:
        ground_truth_indices_str = ground_truth_indices.astype(str)
        true_positives_in_top_k = len(top_k_ranked.intersection(ground_truth_indices_str))
        recall_at_k = true_positives_in_top_k / len(ground_truth_indices_str)

    rws = rank_weighted_score(ranked_ids_only, ground_truth_indices.astype(str))
    return {'recall_at_k': recall_at_k, 'rws': rws, 'top_k': top_k, 'found_in_top_k': true_positives_in_top_k}

# --- Plotting Function ---
def plot_benchmark_results(perf_before, perf_after, perf_gemini, output_path):
    """
    Creates and saves a bar chart comparing the performance metrics.
    """
    labels = ['Initial IForest', 'IForest + Gemini AL', 'Gemini Zero-Shot']
    top_k = perf_before['top_k']
    recalls = [
        perf_before['recall_at_k'],
        perf_after['recall_at_k'],
        perf_gemini['recall_at_k']
    ]
    rws_scores = [
        perf_before['rws'],
        perf_after['rws'],
        perf_gemini['rws']
    ]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, recalls, width, label=f'Recall@{top_k}')
    rects2 = ax.bar(x + width/2, rws_scores, width, label='Rank Weighted Score (RWS)')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Score')
    ax.set_title('Benchmark Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.set_ylim(0, max(max(recalls), max(rws_scores)) * 1.15) # Add space for labels
      # figure out the highest score among recall and RWS
    max_score = max(max(recalls), max(rws_scores))
    # if everything’s zero, default the upper limit to 1.0
    upper = max_score * 1.15 if max_score > 0 else 1.0
    ax.set_ylim(0, upper)

    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plot_path = os.path.join(output_path, 'benchmark_results.png')
    plt.savefig(plot_path)
    print(f"\nBenchmark plot saved to: {plot_path}")
    plt.close(fig)


# --- Main Benchmark Script ---
def main(data_path, solutions_path, output_path):
    """Main execution function."""

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # 1. SETUP
    setup_gemini()

    # 2. LOAD GROUND TRUTH DATA
#     print("Loading ground truth labels...")
#     solutions = pd.read_csv(solutions_path)
#     solutions['GalaxyID'] = solutions['GalaxyID'].astype(str)
#    # true_anomalies_df = solutions[solutions['Class6.1'] > 0.9]
#     true_anomalies_df = solutions[solutions['Class6.1'] >= 0.6]
#     # We use just the ID for ground truth comparison
#     true_anomaly_indices = pd.Index(true_anomalies_df['GalaxyID'].values)
#     print(f"Found {len(true_anomaly_indices)} ground-truth anomalies.")


    # 2. LOAD GROUND TRUTH DATA
    solutions = pd.read_csv(solutions_path, dtype={'GalaxyID': str})
    print("Available columns:", solutions.columns.tolist())

    # Only keep anomalies above your chosen threshold
    anomaly_df = solutions[solutions['Class6.1'] > 0.6]

 
    # 3. RUN THE ASTRONOMALY PIPELINE (PRE-ACTIVE LEARNING)
    print("\n--- Running Benchmark A: ASTRONOMALY with Gemini Oracle ---")
    print("Step 1: Running initial ASTRONOMALY pipeline...")

    dataset = image_reader.ImageThumbnailsDataset(
        directory=data_path,
        transform_function=[image_preprocessing.image_transform_sigma_clipping, image_preprocessing.image_transform_scale],
        display_transform_function=[image_preprocessing.image_transform_scale],
        output_dir=output_path, check_corrupt_data=True
    )
    print(f"Dataset loaded with {len(dataset.index)} images.")

    dataset_ids = set(dataset.index)
    true_anomalies = anomaly_df[anomaly_df['GalaxyID'].isin(dataset_ids)]
    true_anomaly_indices = pd.Index(true_anomalies['GalaxyID'].values)
    print(f"Found {len(true_anomaly_indices)} ground‑truth anomalies in this subset.")

    features = shape_features.EllipseFitFeatures(output_dir=output_path, force_rerun=True, channel=0).run_on_dataset(dataset)

###################### ADDING CNN FEATURES ######################`
#    print("Step 2: Extracting CNN deep features..."
    print("Step 2: Extracting CNN deep features...")
    # Use the CNN_Features class from astronomaly.feature_extraction
    # Ensure the CNN_Features class is imported correctly
    # from astronomaly.feature_extraction import CNN_Features

    from astronomaly.feature_extraction.pretrained_cnn  import  CNN_Features

    
    cnn_stage = CNN_Features(
    model_choice='resnet50',   # or 'resnet18' / 'zoobot'
    output_dir=output_path,
    force_rerun=True
    )
    features = cnn_stage.run_on_dataset(dataset)

   #  features = scaling.FeatureScaler(output_dir=output_path, force_rerun=True).run(features)
    features = scaling.FeatureScaler(
    output_dir=output_path,
    force_rerun=True
    ).run(features)

    initial_scores = isolation_forest.IforestAlgorithm(output_dir=output_path, force_rerun=True, contamination = 0.2).run(features)

    top20_ids = list(initial_scores.sort_values('score', ascending=False).head(20).index)
    print("Top 20 by IForest:", top20_ids)
    print("Ground‑truth anomalies:", list(true_anomaly_indices))

    print("\nEvaluating performance BEFORE active learning...")
    perf_before = evaluate_performance(initial_scores, true_anomaly_indices, top_k=N_FOR_ACTIVE_LEARNING)
    print(f"Initial Recall@{perf_before['top_k']}: {perf_before['recall_at_k']:.4f} ({perf_before['found_in_top_k']}/{len(true_anomaly_indices)})")
    print(f"Initial RWS: {perf_before['rws']:.4f}")

    # 4. ACTIVE LEARNING WITH GEMINI
    print(f"\nStep 2: Performing active learning with Gemini on top {N_FOR_ACTIVE_LEARNING} objects...")
    top_n_anomalies = initial_scores.sort_values('score', ascending=False).head(N_FOR_ACTIVE_LEARNING)
    features_with_labels = pd.concat([features, initial_scores], axis=1)
    features_with_labels['human_label'] = -1

    prompt_active_learning = """
    You are an expert astronomer analyzing galaxy images.
    Please provide a relevance score for the following image on a scale of 0 to 5.
    A score of 5 means the galaxy is highly unusual, anomalous, or interesting (e.g., a merger, strong lenses, disturbed morphology).
    A score of 0 means it is a typical, uninteresting elliptical or spiral galaxy.
    A score of 3 is for mildly interesting objects.
    Please return ONLY the integer score.
    Score:
    """
 

    for i, galaxy_id in enumerate(top_n_anomalies.index):
        image_path = os.path.join(data_path, f"{galaxy_id}.jpg")
        if not os.path.exists(image_path):
            print(f"({i+1}/{N_FOR_ACTIVE_LEARNING}) Thumbnail not found for {galaxy_id}. Skipping.")
            continue

        response_text = get_gemini_response(image_path, prompt_active_learning)
        gemini_score = parse_score_from_response(response_text, max_score=5)

        if gemini_score != -1:
            features_with_labels.loc[image_path, 'human_label'] = gemini_score
            print(f"({i+1}/{N_FOR_ACTIVE_LEARNING}) Object {galaxy_id}: Gemini relevance score = {gemini_score}")
        else:
            print(f"({i+1}/{N_FOR_ACTIVE_LEARNING}) Object {galaxy_id}: Failed to get score from Gemini.")
        time.sleep(2)
        # ── INSERT DEBUG CHECK HERE ──
    print("Labels distribution:", features_with_labels['human_label'].value_counts())
    print("Labelled rows (sample):", features_with_labels.loc[features_with_labels['human_label']>=0].index[:5])
    # ──────────────────────────────
    
   # 5. RUN THE ACTIVE LEARNING MODULE AND EVALUATE

    # features_with_labels = pd.concat([features, initial_scores], axis=1)
    # features_with_labels['human_label'] = -1
    # # … your loop that fills human_label …

    # # Only keep the ones we actually got scores for:
    # labelled = features_with_labels[features_with_labels['human_label'] >= 0].copy()
    # print(f"Collected labels for {len(labelled)} objects.")

    # if labelled.empty:
    #     print("\nNo valid human labels collected. Skipping active learning.")
    #     perf_after = perf_before.copy()
    # else:
    #     # Clean up any missing feature cells before training
    #     labelled.fillna(0, inplace=True)
    #     print(f"After filling NaNs, {len(labelled)} samples remain.")

    #     print(f"\nStep 3: Reranking scores with ASTRONOMY's active learning module on {len(labelled)} labels…")
    #     pipeline_active_learning = human_loop_learning.NeighbourScore(
    #         output_dir=output_path,
    #         force_rerun=True
    #     )
    #     # **Pass only the cleaned 'labelled' subset into run()**
    #     reranked_scores_df = pipeline_active_learning.run(labelled)
    #     final_scores = reranked_scores_df[['trained_score']].rename(columns={'trained_score': 'score'})

    #     print("\nEvaluating performance AFTER active learning…")
    #     perf_after = evaluate_performance(final_scores, true_anomaly_indices,
    #                                     top_k=N_FOR_ACTIVE_LEARNING)
    #     print(f"Final Recall@{perf_after['top_k']}: {perf_after['recall_at_k']:.4f}"
    #         f" ({perf_after['found_in_top_k']}/{len(true_anomaly_indices)})")
    #     print(f"Final RWS: {perf_after['rws']:.4f}")


    # 5. RUN THE ACTIVE LEARNING MODULE AND EVALUATE
    labelled = features_with_labels[features_with_labels['human_label'] >= 0].copy()
    print(f"Collected labels for {len(labelled)} objects.")

    if labelled.empty:
        print("\nNo valid human labels collected. Skipping active learning.")
        perf_after = perf_before.copy()
    else:
        # Fill any missing feature values so Astronomaly's dropna() inside won't remove all rows
        labelled.fillna(0, inplace=True)
        print(f"After filling NaNs, {len(labelled)} samples remain.")

        print(f"\nStep 3: Reranking scores with ASTRONOMY's active learning module on {len(labelled)} labels…")
        pipeline_active_learning = human_loop_learning.NeighbourScore(
            output_dir=output_path,
            force_rerun=True
        )
        # Pass only the cleaned, labelled subset
        reranked_scores_df = pipeline_active_learning.run(labelled)
        final_scores = reranked_scores_df[['trained_score']].rename(columns={'trained_score': 'score'})

        print("\nEvaluating performance AFTER active learning…")
        perf_after = evaluate_performance(
            final_scores,
            true_anomaly_indices,
            top_k=N_FOR_ACTIVE_LEARNING
        )
        print(f"Final Recall@{perf_after['top_k']}: {perf_after['recall_at_k']:.4f}"
              f" ({perf_after['found_in_top_k']}/{len(true_anomaly_indices)})")
        print(f"Final RWS: {perf_after['rws']:.4f}")

    # 6. BENCHMARK GEMINI ZERO-SHOT
    print("\n--- Running Benchmark B: Gemini Zero-Shot Anomaly Detection ---")
    gemini_direct_scores = {}
    prompt_zero_shot = """
    You are an expert astronomer analyzing galaxy images.
    On a scale from 0 to 100, how scientifically interesting or anomalous is this galaxy?
    A score of 100 indicates a very rare or unusual object (e.g., a major merger, a strong gravitational lens, a Voorwerp).
    A score of 0 indicates a completely standard, common elliptical or spiral galaxy.
    Please provide ONLY the integer score.
    Score:
    """

    object_ids_to_score = np.random.choice(dataset.index, size=min(N_FOR_GEMINI_ZERO_SHOT, len(dataset.index)), replace=False)
    # print(f"Scoring {len(object_ids_to_score)} objects directly with Gemini...")

    # for i, image_path in enumerate(object_ids_to_score):
    #     if (i+1) % 10 == 0:
    #         print(f"  ...scored {i+1}/{len(object_ids_to_score)}")

    #     if os.path.exists(image_path):
    #         response_text = get_gemini_response(image_path, prompt_zero_shot)
    #         gemini_score = parse_score_from_response(response_text, max_score=100)
    #         if gemini_score != -1:
    #             gemini_direct_scores[image_path] = gemini_score
    #     else:
    #          print(f"  ...image not found at {image_path}, skipping.")
    #     time.sleep(2)

    # gemini_zero_shot_scores_df = pd.DataFrame.from_dict(gemini_direct_scores, orient='index', columns=['score'])

    print(f"Scoring {len(object_ids_to_score)} objects directly with Gemini...")
    for i, galaxy_id in enumerate(object_ids_to_score):
        image_path = os.path.join(data_path, f"{galaxy_id}.jpg")
        if not os.path.exists(image_path):
            print(f"  ...image not found at {image_path}, skipping.")
            continue

        response_text = get_gemini_response(image_path, prompt_zero_shot)
        gemini_score = parse_score_from_response(response_text, max_score=100)
        if gemini_score != -1:
            gemini_direct_scores[image_path] = gemini_score
        time.sleep(2)

    gemini_zero_shot_scores_df = pd.DataFrame.from_dict(gemini_direct_scores, orient='index', columns=['score'])


    print("\nEvaluating performance of Gemini Zero-Shot ranking...")
    perf_gemini_zero_shot = evaluate_performance(gemini_zero_shot_scores_df, true_anomaly_indices, top_k=N_FOR_ACTIVE_LEARNING)

    # 7. FINAL SUMMARY
    print("\n\n--- BENCHMARK SUMMARY ---")
    print(f"Dataset: Galaxy Zoo Subset ({len(dataset.index)} objects)")
    print(f"Ground Truth Anomalies: {len(true_anomaly_indices)}")
    print("-" * 25)

########################
    print("Total images in dataset:", len(dataset.index))
    print("Total ground‐truth anomalies:", len(anomaly_df))
    print("Sample IDs:", list(anomaly_df['GalaxyID'].head(5)))
#############################

    print("ASTRONOMALY (Initial Isolation Forest):")
    print(f"  Recall@{perf_before['top_k']}: {perf_before['recall_at_k']:.4f}")
    print(f"  Rank Weighted Score (RWS): {perf_before['rws']:.4f}")
    print("-" * 25)
    print(f"ASTRONOMALY (After Active Learning with Gemini on {len(labelled)} objects):")
    print(f"  Recall@{perf_after['top_k']}: {perf_after['recall_at_k']:.4f}")
    print(f"  Rank Weighted Score (RWS): {perf_after['rws']:.4f}")
    print("-" * 25)
    print(f"GEMINI ZERO-SHOT (on a sample of {len(gemini_zero_shot_scores_df)} objects):")
    print(f"  Recall@{perf_gemini_zero_shot['top_k']}: {perf_gemini_zero_shot['recall_at_k']:.4f}")
    print(f"  Rank Weighted Score (RWS): {perf_gemini_zero_shot['rws']:.4f}")
    print("-" * 25)

    # 8. PLOT RESULTS
    plot_benchmark_results(
        perf_before,
        perf_after,
        perf_gemini_zero_shot,
        output_path
    )


if __name__ == "__main__":
    # --- Paths Configuration ---
    # These paths have been set based on the directory you provided.
    # Using forward slashes (/) is recommended for compatibility across operating systems.
    BASE_PATH = "C:/Users/hayda/Downloads/Project_Material/data"

    # 1. Path to the directory containing the Galaxy Zoo JPG images.
    #    Assumes images are in a subdirectory named 'images_gz2'.
    DATA_DIRECTORY = os.path.join(BASE_PATH, "GalaxyZooSubset")

    # 2. Path to the Galaxy Zoo solutions CSV file.
    #    Assumes the file is named 'gz2_solutions.csv'.
    SOLUTIONS_FILE = os.path.join(BASE_PATH, "training_solutions_rev1.csv")

    # 3. Path to the directory where output files will be saved.
    #    A new folder 'astronomaly_output' will be created here.
    OUTPUT_DIRECTORY = os.path.join(BASE_PATH, "astronomaly_output")
    # --- End of Configuration ---

    # Call the main function with the configured paths
    main(
        data_path=DATA_DIRECTORY,
        solutions_path=SOLUTIONS_FILE,
        output_path=OUTPUT_DIRECTORY
    )