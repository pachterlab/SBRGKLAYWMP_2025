#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(zellkonverter)
    library(SingleCellExperiment)
    library(DESeq2)
    library(Matrix)
    library(ggplot2)
})

# Usage:
# Rscript deseq2_simulation_eval.R simulation_data.h5ad output_directory
args <- commandArgs(trailingOnly = TRUE)
script_file <- normalizePath(
    sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)),
    mustWork = FALSE
)
script_dir <- if (length(script_file) > 0) dirname(script_file) else getwd()
h5ad_path <- if (length(args) >= 1) args[[1]] else {
    file.path(script_dir, "results", "simulation_data.h5ad")
}
output_dir <- if (length(args) >= 2) args[[2]] else {
    file.path(script_dir, "results", "deseq2_evaluation")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

alpha <- 0.05
top_n <- 200L

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

# zellkonverter imports AnnData CSR layers as sparse Matrix S4 objects.
# Do not use storage.mode() on them. Keep the sparse payload numeric during
# aggregation; the pseudobulk matrix is rounded and converted to integer below.
if (!inherits(raw_counts, "Matrix")) {
    raw_counts <- Matrix(raw_counts, sparse = TRUE)
}
raw_counts <- as(raw_counts, "CsparseMatrix")

cell_metadata <- as.data.frame(colData(sce))
cell_metadata$strain <- factor(
    as.character(cell_metadata$strain),
    levels = c("StrainA", "StrainB")
)
cell_metadata$cell_type <- factor(
    as.character(cell_metadata$cell_type),
    levels = c("CT1", "CT2", "CT3", "CT4")
)
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
sample_metadata$strain <- factor(
    as.character(sample_metadata$strain),
    levels = c("StrainA", "StrainB")
)
sample_metadata$cell_type <- factor(
    as.character(sample_metadata$cell_type),
    levels = c("CT1", "CT2", "CT3", "CT4")
)

if (!identical(colnames(pseudobulk_counts), rownames(sample_metadata))) {
    stop("Pseudobulk columns and sample metadata rows are not aligned.")
}

dds0 <- DESeqDataSetFromMatrix(
    countData = matrix(
        as.integer(round(as.matrix(pseudobulk_counts))),
        nrow = nrow(pseudobulk_counts),
        ncol = ncol(pseudobulk_counts),
        dimnames = dimnames(pseudobulk_counts)
    ),
    colData = sample_metadata,
    design = ~ strain * cell_type
)

# Wald fit used for the four specificity contrasts.
dds_wald <- DESeq(dds0, test = "Wald", quiet = TRUE)

# -----------------------------------------------------------------------------
# Construct one target-versus-rest contrast for each specificity pattern
# -----------------------------------------------------------------------------
design_grid <- expand.grid(
    strain = c("StrainA", "StrainB"),
    cell_type = c("CT1", "CT2", "CT3", "CT4"),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
)
design_grid$strain <- factor(
    design_grid$strain,
    levels = c("StrainA", "StrainB")
)
design_grid$cell_type <- factor(
    design_grid$cell_type,
    levels = c("CT1", "CT2", "CT3", "CT4")
)
design_grid$group <- paste(
    design_grid$strain,
    design_grid$cell_type,
    sep = ":"
)

grid_matrix <- model.matrix(~ strain * cell_type, data = design_grid)
if (ncol(grid_matrix) != length(resultsNames(dds_wald))) {
    stop(
        "The numeric contrast length does not match resultsNames(dds).\n",
        "Model-matrix columns: ", paste(colnames(grid_matrix), collapse = ", "),
        "\nDESeq2 coefficients: ", paste(resultsNames(dds_wald), collapse = ", ")
    )
}

target_vs_rest <- function(target_groups) {
    target <- design_grid$group %in% target_groups
    if (!any(target) || all(target)) {
        stop("Invalid target groups: ", paste(target_groups, collapse = ", "))
    }
    colMeans(grid_matrix[target, , drop = FALSE]) -
        colMeans(grid_matrix[!target, , drop = FALSE])
}

contrast_definitions <- list(
    "Category 1: CT1" = c("StrainA:CT1", "StrainB:CT1"),
    "Category 2: CT1 + CT2" = c(
        "StrainA:CT1", "StrainA:CT2",
        "StrainB:CT1", "StrainB:CT2"
    ),
    "Category 4: StrainA:CT1" = "StrainA:CT1",
    "Category 5: StrainA:CT1 + StrainB:CT2" = c(
        "StrainA:CT1", "StrainB:CT2"
    )
)

extract_wald_top <- function(category, target_groups) {
    contrast_vector <- target_vs_rest(target_groups)
    result <- results(
        dds_wald,
        contrast = contrast_vector,
        alpha = alpha,
        independentFiltering = FALSE
    )
    table <- as.data.frame(result)
    table$gene <- rownames(table)
    table$category <- category
    table$eligible <- (
        !is.na(table$padj) &
        table$padj < alpha &
        !is.na(table$log2FoldChange) &
        table$log2FoldChange > 0 &
        !is.na(table$stat)
    )

    eligible <- table[table$eligible, , drop = FALSE]
    eligible <- eligible[order(-eligible$stat, eligible$gene), , drop = FALSE]
    eligible$eligible_rank <- seq_len(nrow(eligible))
    top <- head(eligible, top_n)

    safe_name <- gsub("[^A-Za-z0-9]+", "_", category)
    write.csv(
        table,
        file.path(output_dir, paste0(safe_name, "_all_genes.csv")),
        row.names = FALSE
    )
    write.csv(
        top,
        file.path(output_dir, paste0(safe_name, "_top200.csv")),
        row.names = FALSE
    )
    top
}

top_lists <- list()
for (category in names(contrast_definitions)) {
    top_lists[[category]] <- extract_wald_top(
        category,
        contrast_definitions[[category]]
    )
}

# -----------------------------------------------------------------------------
# Housekeeping: no detectable strain/cell-type effect, ranked by abundance
# -----------------------------------------------------------------------------
dds_lrt <- DESeq(
    dds0,
    test = "LRT",
    reduced = ~ 1,
    quiet = TRUE
)
housekeeping_result <- results(
    dds_lrt,
    alpha = alpha,
    independentFiltering = FALSE
)
housekeeping <- as.data.frame(housekeeping_result)
housekeeping$gene <- rownames(housekeeping)
housekeeping$category <- "Category 3: Housekeeping"
housekeeping$eligible <- (
    !is.na(housekeeping$padj) &
    housekeeping$padj >= alpha &
    !is.na(housekeeping$baseMean)
)
housekeeping_eligible <- housekeeping[
    housekeeping$eligible,
    ,
    drop = FALSE
]
housekeeping_eligible <- housekeeping_eligible[
    order(-housekeeping_eligible$baseMean, housekeeping_eligible$gene),
    ,
    drop = FALSE
]
housekeeping_eligible$eligible_rank <- seq_len(nrow(housekeeping_eligible))
top_lists[["Category 3: Housekeeping"]] <- head(
    housekeeping_eligible,
    top_n
)

write.csv(
    housekeeping,
    file.path(output_dir, "Category_3_Housekeeping_all_genes.csv"),
    row.names = FALSE
)
write.csv(
    top_lists[["Category 3: Housekeeping"]],
    file.path(output_dir, "Category_3_Housekeeping_top200.csv"),
    row.names = FALSE
)

# -----------------------------------------------------------------------------
# Evaluate recovery of the known 200-gene truth sets
# -----------------------------------------------------------------------------
evaluate_top <- function(category, top) {
    bounds <- truth_ranges[[category]]
    gene_index <- suppressWarnings(
        as.integer(sub("^gene_", "", top$gene))
    )
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

category_order <- names(truth_ranges)
evaluation <- do.call(
    rbind,
    lapply(category_order, function(category) {
        evaluate_top(category, top_lists[[category]])
    })
)
rownames(evaluation) <- NULL
write.csv(
    evaluation,
    file.path(output_dir, "deseq2_recovery_results.csv"),
    row.names = FALSE
)

combined_top <- do.call(
    rbind,
    lapply(category_order, function(category) top_lists[[category]])
)
write.csv(
    combined_top,
    file.path(output_dir, "deseq2_top_200.csv"),
    row.names = FALSE
)

# -----------------------------------------------------------------------------
# Bar plot matching the EMBER evaluation
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

plot <- ggplot(evaluation, aes(x = short_label, y = recovered, fill = short_label)) +
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
        title = "DESeq2 recovery of simulated specificity patterns",
        x = "Simulated specificity pattern",
        y = "True genes recovered in DESeq2 top 200"
    ) +
    theme_classic(base_size = 14) +
    theme(
        axis.text.x = element_text(angle = 20, hjust = 1),
        plot.title = element_text(hjust = 0.5)
    )

ggsave(
    file.path(output_dir, "deseq2_top200_recovery.png"),
    plot,
    width = 10,
    height = 6,
    dpi = 300
)
ggsave(
    file.path(output_dir, "deseq2_top200_recovery.pdf"),
    plot,
    width = 10,
    height = 6
)

print(evaluation)
cat("\nSaved DESeq2 evaluation to:", output_dir, "\n")
