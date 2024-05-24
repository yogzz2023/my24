import numpy as np
from scipy.stats import chi2

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)

# Predefined tracks and reports in 3D
tracks = np.array([
    [6, 6, 10],
    [15, 15, 10],
    [7, 7, 10]
])

reports = np.array([
    [7, 7, 10],
    [16, 16, 10],
    [8, 8, 10],
    [80, 80, 80]
])

# Chi-squared gating threshold for 95% confidence interval
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = x - y
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Covariance matrix of the measurement errors (assumed to be identity for simplicity)
cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

# Perform residual error check using Chi-squared gating
association_list = []
for i, track in enumerate(tracks):
    for j, report in enumerate(reports):
        distance = mahalanobis_distance(track, report, cov_inv)
        if distance < chi2_threshold:
            association_list.append((i, j))

# Print association list
print("Association List (Track Index, Report Index):")
for assoc in association_list:
    print(assoc)

# Clustering reports and tracks based on associations
clusters = []
while association_list:
    cluster_tracks = set()
    cluster_reports = set()
    stack = [association_list.pop(0)]
    while stack:
        track_idx, report_idx = stack.pop()
        cluster_tracks.add(track_idx)
        cluster_reports.add(report_idx)
        new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
        for assoc in new_assoc:
            if assoc not in stack:
                stack.append(assoc)
        association_list = [assoc for assoc in association_list if assoc not in new_assoc]
    clusters.append((list(cluster_tracks), list(cluster_reports)))

# Print clusters
print("\nClusters (Tracks, Reports):")
for cluster in clusters:
    print(cluster)

# Hypothesis generation for each cluster
def generate_hypotheses(tracks, reports):
    if tracks.size == 0 or reports.size == 0:
        return []

    hypotheses = []
    num_tracks = len(tracks)
    num_reports = len(reports)

    def is_valid_hypothesis(hypothesis):
        report_set = set()
        for report_idx in hypothesis:
            if report_idx in report_set:
                return False
            report_set.add(report_idx)
        return True

    # Only generate hypotheses where each track is associated with a report
    for hypothesis in np.ndindex(*(num_reports for _ in range(num_tracks))):
        if is_valid_hypothesis(hypothesis):
            hypotheses.append(hypothesis)

    print("\nGenerated Hypotheses:")
    for hypothesis in hypotheses:
        hyp_details = []
        for track_idx, report_idx in enumerate(hypothesis):
            hyp_details.append((track_idx, report_idx, reports[report_idx].tolist()))
        print(hyp_details)

    return hypotheses

# Calculate probabilities for each hypothesis
def calculate_probabilities(hypotheses, tracks, reports):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in enumerate(hypothesis):
            distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
            prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize
    return probabilities

# Process each cluster and generate hypotheses
for track_idxs, report_idxs in clusters:
    cluster_tracks = tracks[track_idxs]
    cluster_reports = reports[report_idxs]
    hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
    probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports)
    print("\nCluster Hypotheses and Probabilities:")
    for hypothesis, probability in zip(hypotheses, probabilities):
        hyp_details = []
        for track_idx, report_idx in enumerate(hypothesis):
            hyp_details.append((track_idxs[track_idx], report_idxs[report_idx], cluster_reports[report_idx].tolist()))
        print(f"Hypothesis: {hyp_details}, Probability: {probability:.4f}")
