# Put wdbc.data in the folder where you run this script.
import numpy as np


# Settings
DATA_FILE = "wdbc.data"
N_FOLDS = 5
PCA_VARIANCE_THRESHOLD = 0.80
RANDOM_SEED = 42

def loadWdbc(filename):
    data = np.genfromtxt(filename, delimiter=",", dtype=str)
    featureData = data[:, 2:]
    X = featureData.astype(float)
    labels = data[:, 1]
    isMalignant = labels == "M"
    y = isMalignant.astype(int)  # M = 1, B = 0
    return X, y


def makeFolds(y, nFolds=5, seed=42):
    rng = np.random.default_rng(seed)

    benignIndices = np.where(y == 0)[0]
    malignantIndices = np.where(y == 1)[0]

    rng.shuffle(benignIndices)
    rng.shuffle(malignantIndices)

    benign_splits = np.array_split(benignIndices, nFolds)
    malignant_splits = np.array_split(malignantIndices, nFolds)

    folds = []
    for fold_index in range(nFolds):
        benign_group = benign_splits[fold_index]
        malignant_group = malignant_splits[fold_index]
        fold = np.concatenate((benign_group, malignant_group))
        rng.shuffle(fold)
        folds.append(fold)

    return folds


def trainNMC(X_train, y_train):
    benign_rows = y_train == 0
    malignant_rows = y_train == 1

    benign_data = X_train[benign_rows]
    malignant_data = X_train[malignant_rows]

    class_0_mean = np.mean(benign_data, axis=0)
    class_1_mean = np.mean(malignant_data, axis=0)

    means = [class_0_mean, class_1_mean]
    class_means = np.array(means)
    return class_means


def predictNMC(X, class_means):
    predictions = []

    for sample in X:
        difference_0 = sample - class_means[0]
        difference_1 = sample - class_means[1]

        distance_0 = np.sum(difference_0 ** 2)
        distance_1 = np.sum(difference_1 ** 2)

        # Choose class 0 if both distances are equal.
        if distance_0 <= distance_1:
            predictions.append(0)
        else:
            predictions.append(1)

    return np.array(predictions, dtype=int)


def accuracy_percent(y_true, y_pred):
    correct_predictions = y_true == y_pred
    correct_count = np.sum(correct_predictions)
    total_count = len(y_true)
    accuracy = correct_count / total_count * 100
    return accuracy


def fit_pca(X_train, variance_threshold=0.80):
    train_mean = np.mean(X_train, axis=0)
    X_centered = X_train - train_mean

    sample_count = X_centered.shape[0]
    transposed_data = X_centered.T
    covariance_matrix = np.matmul(transposed_data, X_centered)
    covariance_matrix = covariance_matrix / (sample_count - 1)

    # Find the PCA directions and their variances.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    order = np.argsort(eigenvalues)
    order = np.flip(order)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Rounding can produce tiny negative eigenvalues.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total_variance = np.sum(eigenvalues)
    if total_variance == 0.0:
        raise ValueError("PCA failed because the training data has zero variance.")

    explained_ratio = eigenvalues / total_variance
    cumulative_ratio = np.cumsum(explained_ratio)

    selected_count = 0
    for ratio in cumulative_ratio:
        selected_count = selected_count + 1
        if ratio >= variance_threshold:
            break
    selected_components = eigenvectors[:, :selected_count]

    return train_mean, selected_components, cumulative_ratio, selected_count


def transform_pca(X, train_mean, components):
    centered_data = X - train_mean
    transformed_data = np.matmul(centered_data, components)
    return transformed_data


# Simple SVG plot writer (no plotting library required)
def save_pca_svg(cumulative_curves, filename="pca_explained_variance.svg"):
    width = 900
    height = 600
    left = 90
    right = 35
    top = 45
    bottom = 80

    plot_width = width - left - right
    plot_height = height - top - bottom
    n_components = cumulative_curves.shape[1]

    def x_pixel(component_number):
        if n_components == 1:
            return left
        return left + (component_number - 1) * plot_width / (n_components - 1)

    def y_pixel(value):
        return top + (1.0 - value) * plot_height

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="white"/>')

    # Title
    lines.append(
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        'font-family="Arial" font-size="20">Cumulative Proportion of Variance Explained</text>'
    )

    # Axes
    x0 = left
    y0 = top + plot_height
    lines.append(
        f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="black" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{x0}" y1="{y0}" x2="{left + plot_width}" y2="{y0}" '
        'stroke="black" stroke-width="2"/>'
    )

    # Horizontal grid/ticks: 0.0 to 1.0
    for tick in range(0, 11):
        value = tick / 10.0
        y = y_pixel(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{value:.1f}</text>'
        )

    # Vertical x ticks at selected component numbers.
    tick_count = min(10, n_components)
    tick_positions = np.linspace(1, n_components, tick_count)
    tick_positions = np.rint(tick_positions)
    tick_positions = tick_positions.astype(int)
    tick_positions = np.unique(tick_positions)
    for component_number in tick_positions:
        x = x_pixel(int(component_number))
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{y0}" '
            'stroke="#eeeeee" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{y0 + 22}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{int(component_number)}</text>'
        )

    # 80% reference line
    y80 = y_pixel(0.80)
    lines.append(
        f'<line x1="{left}" y1="{y80:.2f}" x2="{left + plot_width}" y2="{y80:.2f}" '
        'stroke="#777777" stroke-width="1.5" stroke-dasharray="6,5"/>'
    )
    lines.append(
        f'<text x="{left + plot_width - 4}" y="{y80 - 6:.2f}" text-anchor="end" '
        'font-family="Arial" font-size="12">80%</text>'
    )

    # Curves
    for fold_index in range(cumulative_curves.shape[0]):
        points = []
        for j in range(n_components):
            x = x_pixel(j + 1)
            y = y_pixel(float(cumulative_curves[fold_index, j]))
            points.append(f"{x:.2f},{y:.2f}")

        color_index = fold_index % len(colors)
        color = colors[color_index]
        curve_points = " ".join(points)
        lines.append(
            f'<polyline points="{curve_points}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>'
        )

        legend_x = left + plot_width - 125
        legend_y = top + 24 + fold_index * 23
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-family="Arial" '
            f'font-size="12">Fold {fold_index + 1}</text>'
        )

    # Axis labels
    lines.append(
        f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" '
        'font-family="Arial" font-size="15">Number of Principal Components</text>'
    )
    lines.append(
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" '
        'font-family="Arial" font-size="15" '
        f'transform="rotate(-90 24 {top + plot_height / 2})">Cumulative Proportion of Variance Explained</text>'
    )

    lines.append("</svg>")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# Main experiment
def main():
    X, y = loadWdbc(DATA_FILE)

    if X.shape[0] != 569 or X.shape[1] != 30:
        print(
            "Warning: the assignment describes 569 samples and 30 input features, "
            f"but this file contains {X.shape[0]} samples and {X.shape[1]} features."
        )

    folds = makeFolds(y, N_FOLDS, RANDOM_SEED)

    nmc_accuracies = np.zeros(N_FOLDS)
    pca_nmc_accuracies = np.zeros(N_FOLDS)
    train_sizes = np.zeros(N_FOLDS, dtype=int)
    validation_sizes = np.zeros(N_FOLDS, dtype=int)
    selected_pc_counts = np.zeros(N_FOLDS, dtype=int)
    cumulative_curves = np.zeros((N_FOLDS, X.shape[1]))

    all_indices = np.arange(X.shape[0])

    print("\n5-Fold Cross-Validation Results")
    print("-" * 86)
    print(
        f"{'Fold':<6}{'Train/Validation':<20}{'NMC Accuracy (%)':<20}"
        f"{'PCA-NMC Accuracy (%)':<24}{'PCs':<6}"
    )
    print("-" * 86)

    for fold_index in range(N_FOLDS):
        validation_indices = folds[fold_index]

        is_validation = np.zeros(X.shape[0], dtype=bool)
        is_validation[validation_indices] = True
        is_training = is_validation == False
        train_indices = all_indices[is_training]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        train_sizes[fold_index] = X_train.shape[0]
        validation_sizes[fold_index] = X_validation.shape[0]

        # Step (a): NMC on original features.
        original_class_means = trainNMC(X_train, y_train)
        original_predictions = predictNMC(X_validation, original_class_means)
        nmc_accuracy = accuracy_percent(y_validation, original_predictions)
        nmc_accuracies[fold_index] = nmc_accuracy

        # Steps (b)-(c): fit PCA only on training data and transform both sets.
        train_mean, components, cumulative_ratio, selected_count = fit_pca(
            X_train, PCA_VARIANCE_THRESHOLD
        )
        cumulative_curves[fold_index] = cumulative_ratio
        selected_pc_counts[fold_index] = selected_count

        X_train_pca = transform_pca(X_train, train_mean, components)
        X_validation_pca = transform_pca(X_validation, train_mean, components)

        # Steps (d)-(e): NMC in PCA space and validation accuracy.
        pca_class_means = trainNMC(X_train_pca, y_train)
        pca_predictions = predictNMC(X_validation_pca, pca_class_means)
        pca_accuracy = accuracy_percent(y_validation, pca_predictions)
        pca_nmc_accuracies[fold_index] = pca_accuracy

        print(
            f"{fold_index + 1:<6}"
            f"{str(X_train.shape[0]) + '/' + str(X_validation.shape[0]):<20}"
            f"{nmc_accuracy:<20.2f}"
            f"{pca_accuracy:<24.2f}"
            f"{selected_count:<6d}"
        )

    print("-" * 86)

    nmc_mean = np.mean(nmc_accuracies)
    nmc_std = np.std(nmc_accuracies)
    pca_nmc_mean = np.mean(pca_nmc_accuracies)
    pca_nmc_std = np.std(pca_nmc_accuracies)

    print(f"NMC mean +/- std:     {nmc_mean:.2f} +/- {nmc_std:.2f}%")
    print(f"PCA-NMC mean +/- std: {pca_nmc_mean:.2f} +/- {pca_nmc_std:.2f}%")
    print("PCs retained in folds:", selected_pc_counts)

    results = []
    for fold_index in range(N_FOLDS):
        row = [
            fold_index + 1,
            train_sizes[fold_index],
            validation_sizes[fold_index],
            nmc_accuracies[fold_index],
            pca_nmc_accuracies[fold_index],
            selected_pc_counts[fold_index],
        ]
        results.append(row)

    results = np.array(results)
    np.savetxt(
        "assignment2_results.csv",
        results,
        delimiter=",",
        header=(
            "fold,training_samples,validation_samples,nmc_accuracy_percent,"
            "pca_nmc_accuracy_percent,selected_principal_components"
        ),
        comments="",
        fmt=["%d", "%d", "%d", "%.6f", "%.6f", "%d"],
    )

    # Save cumulative explained-variance values for all five folds.
    variance_output = []
    for component_index in range(X.shape[1]):
        row = [component_index + 1]
        for fold_index in range(N_FOLDS):
            ratio = cumulative_curves[fold_index, component_index]
            row.append(ratio)
        variance_output.append(row)

    variance_output = np.array(variance_output)
    np.savetxt(
        "pca_explained_variance.csv",
        variance_output,
        delimiter=",",
        header="component,fold1,fold2,fold3,fold4,fold5",
        comments="",
        fmt="%.8f",
    )

    # Produce the requested five-curve PCA figure without matplotlib.
    save_pca_svg(cumulative_curves)

    print("\nSaved:")
    print("  assignment2_results.csv")
    print("  pca_explained_variance.csv")
    print("  pca_explained_variance.svg")


if __name__ == "__main__":
    main()
