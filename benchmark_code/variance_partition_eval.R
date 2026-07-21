#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(zellkonverter)
    library(SingleCellExperiment)
    library(Matrix)
    library(edgeR)
    library(variancePartition)
    library(BiocParallel)
    library(ggplot2)
})

# Usage:
# Rscript variance_partition_simulation_eval.R input.h5ad output_dir [n_cores]
args <- commandArgs(trailingOnly = TRUE)
script_file <- normalizePath(
    sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)),
    mustWork = FALSE
)
script_dir <- if (length(script_file) > 0) dirname(script_file) else getwd()
h5ad_path <- if (length(args) >= 1) args[[1]] else {
    file.path(script_dir, "results", "simulation_data_r_compatible.h5ad")
}
output_dir <- if (length(args) >= 2) args[[2]] else {
    file.path(script_dir, "results", "variance_partition_evaluation")
}
n_cores <- if (length(args) >= 3) as.integer(args[[3]]) else 4L
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

top_n <- 200L
housekeeping_max_biological_variance <- 0.05

truth_ranges <- list(
    "Category 1: CT1" = c(0L, 199L),
    "Category 2: CT1 + CT2" = c(200L, 399L),
    "Category 3: Housekeeping" = c(400L, 599L),
    "Category 4: StrainA:CT1" = c(600L, 799L),
    "Category 5: StrainA:CT1 + StrainB:CT2" = c(800L, 999L)
)

# -----------------------------------------------------------------------------
# Read raw counts and pseudobulk by mouse x cell type
# -----------------------------------------------------------------------------
sce <- readH5AD(h5ad_path)
if (!"raw_counts" %in% assayNames(sce)) {
    stop("The h5ad does not contain an assay/layer named 'raw_counts'.")
}

required_obs <- c("mouse_id", "strain", "cell_type")
missing_obs <- setdiff(required_obs, colnames(colData(sce)))
if (length(missing_obs) > 0) {
    stop("Missing observation columns: ", paste(missing_obs, collapse = ", "))
}

raw_counts <- assay(sce, "raw_counts")
if (!inherits(raw_counts, "Matrix")) {
    raw_counts <- Matrix(raw_counts, sparse = TRUE)
}
raw_counts <- as(raw_counts, "CsparseMatrix")

cell_metadata <- as.data.frame(colData(sce))
cell_metadata$strain <- factor(as.character(cell_metadata$strain))
cell_metadata$cell_type <- factor(as.character(cell_metadata$cell_type))
cell_metadata$sample_key <- paste(
    cell_metadata$mouse_id,
    cell_metadata$cell_type,
    sep = "__"
)

sample_factor <- factor(
    cell_metadata$sample_key,
    levels = unique(cell_metadata$sample_key)
)
aggregation_matrix <- sparse.model.matrix(~ 0 + sample_factor)
pseudobulk_counts <- raw_counts %*% aggregation_matrix
colnames(pseudobulk_counts) <- levels(sample_factor)

first_cell <- match(levels(sample_factor), cell_metadata$sample_key)
sample_metadata <- cell_metadata[
    first_cell,
    c("mouse_id", "strain", "cell_type"),
    drop = FALSE
]
rownames(sample_metadata) <- levels(sample_factor)
sample_metadata$mouse_id <- factor(as.character(sample_metadata$mouse_id))
sample_metadata$strain <- factor(as.character(sample_metadata$strain))
sample_metadata$cell_type <- factor(as.character(sample_metadata$cell_type))
sample_metadata$strain_celltype <- interaction(
    sample_metadata$strain,
    sample_metadata$cell_type,
    sep = ":",
    drop = TRUE
)

if (!identical(colnames(pseudobulk_counts), rownames(sample_metadata))) {
    stop("Pseudobulk columns and sample metadata rows are not aligned.")
}

count_matrix <- matrix(
    as.integer(round(as.matrix(pseudobulk_counts))),
    nrow = nrow(pseudobulk_counts),
    ncol = ncol(pseudobulk_counts),
    dimnames = dimnames(pseudobulk_counts)
)

# -----------------------------------------------------------------------------
# Fit variancePartition model
# -----------------------------------------------------------------------------
dge <- DGEList(counts = count_matrix)
dge <- calcNormFactors(dge)

formula <- ~
    (1 | strain) +
    (1 | cell_type) +
    (1 | strain_celltype) +
    (1 | mouse_id)

if (n_cores > 1L && .Platform$OS.type != "windows") {
    parallel_param <- MulticoreParam(n_cores)
} else {
    parallel_param <- SerialParam()
}

voom_object <- voomWithDreamWeights(
    dge,
    formula,
    sample_metadata,
    BPPARAM = parallel_param
)

variance_fractions <- fitExtractVarPartModel(
    voom_object,
    formula,
    sample_metadata,
    REML = FALSE,
    BPPARAM = parallel_param
)

variance_table <- as.data.frame(variance_fractions)
variance_table$gene <- rownames(variance_table)
variance_table$mean_logCPM <- rowMeans(voom_object$E)

required_components <- c(
    "strain", "cell_type", "strain_celltype", "mouse_id", "Residuals"
)
missing_components <- setdiff(required_components, colnames(variance_table))
if (length(missing_components) > 0) {
    stop(
        "Missing variance components: ",
        paste(missing_components, collapse = ", "),
        "\nAvailable columns: ",
        paste(colnames(variance_table), collapse = ", ")
    )
}

write.csv(
    variance_table,
    file.path(output_dir, "variance_partition_all_genes.csv"),
    row.names = FALSE
)

# -----------------------------------------------------------------------------
# Extract top 200 for each pattern
# -----------------------------------------------------------------------------
rank_component <- function(category, component) {
    table <- variance_table
    table$category <- category
    table$ranking_component <- component
    table$ranking_value <- table[[component]]
    table <- table[
        !is.na(table$ranking_value),
        ,
        drop = FALSE
    ]
    table <- table[
        order(-table$ranking_value, table$gene),
        ,
        drop = FALSE
    ]
    table$rank <- seq_len(nrow(table))
    head(table, top_n)
}

# variancePartition returns one variance fraction per source. It does not say
# which cell type or which joint category generated that variance. Therefore
# Categories 1 and 2 necessarily share a ranking, as do Categories 4 and 5.
top_lists <- list(
    "Category 1: CT1" = rank_component(
        "Category 1: CT1", "cell_type"
    ),
    "Category 2: CT1 + CT2" = rank_component(
        "Category 2: CT1 + CT2", "cell_type"
    ),
    "Category 4: StrainA:CT1" = rank_component(
        "Category 4: StrainA:CT1", "strain_celltype"
    ),
    "Category 5: StrainA:CT1 + StrainB:CT2" = rank_component(
        "Category 5: StrainA:CT1 + StrainB:CT2", "strain_celltype"
    )
)

# Housekeeping genes should be highly expressed but have little biological
# variance attributable to strain, cell type, or their joint partition.
housekeeping <- variance_table
housekeeping$biological_variance <- (
    housekeeping$strain +
    housekeeping$cell_type +
    housekeeping$strain_celltype
)
housekeeping$category <- "Category 3: Housekeeping"
housekeeping$ranking_component <- "mean_logCPM"
housekeeping$ranking_value <- housekeeping$mean_logCPM
housekeeping <- housekeeping[
    !is.na(housekeeping$biological_variance) &
    housekeeping$biological_variance <= housekeeping_max_biological_variance &
    !is.na(housekeeping$mean_logCPM),
    ,
    drop = FALSE
]
housekeeping <- housekeeping[
    order(-housekeeping$mean_logCPM, housekeeping$gene),
    ,
    drop = FALSE
]
housekeeping$rank <- seq_len(nrow(housekeeping))
top_lists[["Category 3: Housekeeping"]] <- head(housekeeping, top_n)

category_order <- names(truth_ranges)
for (category in category_order) {
    safe_name <- gsub("[^A-Za-z0-9]+", "_", category)
    write.csv(
        top_lists[[category]],
        file.path(output_dir, paste0(safe_name, "_top200.csv")),
        row.names = FALSE
    )
}

# -----------------------------------------------------------------------------
# Evaluate recovery
# -----------------------------------------------------------------------------
evaluate_top <- function(category, top) {
    bounds <- truth_ranges[[category]]
    gene_index <- suppressWarnings(as.integer(sub("^gene_", "", top$gene)))
    recovered <- sum(
        !is.na(gene_index) &
        gene_index >= bounds[[1]] &
        gene_index <= bounds[[2]]
    )
    selected <- nrow(top)
    random_expected <- selected * 200 / 3000

    data.frame(
        category = category,
        n_selected = selected,
        recovered = recovered,
        out_of = 200L,
        precision_at_200 = if (selected > 0) recovered / selected else NA_real_,
        recall_at_200 = recovered / 200,
        false_positives = selected - recovered,
        fold_enrichment_over_random = if (random_expected > 0) {
            recovered / random_expected
        } else {
            NA_real_
        },
        hypergeometric_pvalue = phyper(
            recovered - 1,
            200,
            2800,
            selected,
            lower.tail = FALSE
        )
    )
}

evaluation <- do.call(
    rbind,
    lapply(category_order, function(category) {
        evaluate_top(category, top_lists[[category]])
    })
)
rownames(evaluation) <- NULL
write.csv(
    evaluation,
    file.path(output_dir, "variance_partition_recovery_results.csv"),
    row.names = FALSE
)

rbind_fill <- function(frames) {
    all_columns <- unique(unlist(lapply(frames, colnames)))
    aligned <- lapply(frames, function(frame) {
        missing <- setdiff(all_columns, colnames(frame))
        for (column in missing) {
            frame[[column]] <- NA
        }
        frame[, all_columns, drop = FALSE]
    })
    do.call(rbind, aligned)
}

combined_top <- rbind_fill(
    lapply(category_order, function(category) top_lists[[category]])
)
write.csv(
    combined_top,
    file.path(output_dir, "variance_partition_top_200.csv"),
    row.names = FALSE
)

# -----------------------------------------------------------------------------
# Recovery bar plot
# -----------------------------------------------------------------------------
evaluation$short_label <- factor(
    c("1 CT", "2 CTs", "Housekeeping", "1 CT, 1 strain", "Strain switch"),
    levels = c("1 CT", "2 CTs", "Housekeeping", "1 CT, 1 strain", "Strain switch")
)
evaluation$bar_label <- paste0(evaluation$recovered, "/200")
random_expectation <- top_n * 200 / 3000

colors <- c(
    "1 CT" = "#3B82F6",
    "2 CTs" = "#6366F1",
    "Housekeeping" = "#10B981",
    "1 CT, 1 strain" = "#F59E0B",
    "Strain switch" = "#EF4444"
)

plot <- ggplot(evaluation, aes(short_label, recovered, fill = short_label)) +
    geom_col(color = "black", linewidth = 0.5, width = 0.8) +
    geom_text(aes(label = bar_label), vjust = -0.6, size = 4.5) +
    geom_hline(
        yintercept = random_expectation,
        linetype = "dashed",
        linewidth = 0.8
    ) +
    annotate(
        "text",
        x = 5.45,
        y = random_expectation + 5,
        label = sprintf("Random expectation (%.1f/200)", random_expectation),
        hjust = 1,
        size = 4
    ) +
    scale_fill_manual(values = colors, guide = "none") +
    scale_y_continuous(limits = c(0, 210), expand = expansion(mult = c(0, 0))) +
    labs(
        title = "variancePartition recovery of simulated specificity patterns",
        x = "Simulated specificity pattern",
        y = "True genes recovered in variancePartition top 200"
    ) +
    theme_classic(base_size = 14) +
    theme(
        axis.text.x = element_text(angle = 20, hjust = 1),
        plot.title = element_text(hjust = 0.5)
    )

ggsave(
    file.path(output_dir, "variance_partition_top200_recovery.png"),
    plot,
    width = 10,
    height = 6,
    dpi = 300
)
ggsave(
    file.path(output_dir, "variance_partition_top200_recovery.pdf"),
    plot,
    width = 10,
    height = 6
)

print(evaluation)
cat("\nSaved variancePartition evaluation to:", output_dir, "\n")
